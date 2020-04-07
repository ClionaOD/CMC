"""
Train CMC with AlexNet
"""
from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket
import numpy as np

import tensorboard_logger as tb_logger

from torchvision import transforms, datasets, models
from dataset import RGB2Lab, RGB2YCbCr
from util import adjust_learning_rate, AverageMeter

from models.alexnet import MyAlexNetCMC
from models.alexnet import TemporalAlexNetCMC #COD 20/02/07
from models.resnet import MyResNetsCMC
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from dataset import ImageFolderInstance
from dataset import twoImageFolderInstance #COD 20/02/07

try:
    from apex import amp, optimizers
except ImportError:
    pass
"""
TODO: python 3.6 ModuleNotFoundError
"""


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3'])
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet'])

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr', 'temporal']) #COD 20/02/06 added temporal

    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    # range for timelag to test (COD 20/02/06)
    parser.add_argument('--time_lag', type=int, default=100, help='number of 1 second frames to lag by')

    # use ImageNet pretrained AlexNet
    parser.add_argument('--pretrained', type=str, default=False, help='whether to start from pretrained AlexNet')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = 'memory_{}_{}_{}_lr_{}_decay_{}_bsz_{}_sec_{}'.format(opt.method, opt.nce_k, opt.model, opt.learning_rate,
                                                                    opt.weight_decay, opt.batch_size, opt.time_lag)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)
    if opt.pretrained:
        opt.model_name = 'pretrained_{}'.format(opt.model_name)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt


def get_train_loader(args):
    """get the train loader"""
    data_folder = os.path.join(args.data_folder, 'train')

    if args.view == 'Lab':
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = RGB2Lab()
    elif args.view == 'YCbCr':
        mean = [116.151, 121.080, 132.342]
        std = [109.500, 111.855, 111.964]
        color_transfer = RGB2YCbCr()
    elif args.view == 'temporal':                                                       #Use Lab for comparison 
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = RGB2Lab()
    else:
        raise NotImplemented('view not implemented {}'.format(args.view))
    
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        color_transfer,
        transforms.ToTensor(),
        normalize,
    ])

    #COD 20/02/07 - include train_dataset for loading two images
    if not args.view == 'temporal':
        train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    else:
        train_dataset = twoImageFolderInstance(data_folder, time_lag=args.time_lag, transform=train_transform)
    train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        if not args.view == 'temporal':             #COD 20/02/07 include two full alexnets for the temporal view
            model = MyAlexNetCMC(args.feat_dim)     
        else:
            if not args.pretrained:
                model = TemporalAlexNetCMC(args.feat_dim)
            else:
                model = MyAlexNetCMC()
    elif args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax) 
    criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion_ab, criterion_l


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, contrast, criterion_l, criterion_ab, optimizer, opt):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()

    end = time.time()
    if not opt.view == 'temporal':
        for idx, (inputs, _, index) in enumerate(train_loader):
            data_time.update(time.time() - end)

            bsz = inputs.size(0)
            inputs = inputs.float()
            if torch.cuda.is_available():
                index = index.cuda(non_blocking=True)
                inputs = inputs.cuda()

            # ===================forward=====================
            feat_l, feat_ab = model(inputs)
            out_l, out_ab = contrast(feat_l, feat_ab, index)

            l_loss = criterion_l(out_l)
            ab_loss = criterion_ab(out_ab)
            l_prob = out_l[:, 0].mean()
            ab_prob = out_ab[:, 0].mean()

            loss = l_loss + ab_loss

            # ===================backward=====================
            optimizer.zero_grad()
            if opt.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # ===================meters=====================
            losses.update(loss.item(), bsz)
            l_loss_meter.update(l_loss.item(), bsz)
            l_prob_meter.update(l_prob.item(), bsz)
            ab_loss_meter.update(ab_loss.item(), bsz)
            ab_prob_meter.update(ab_prob.item(), bsz)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                    'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, lprobs=l_prob_meter,
                    abprobs=ab_prob_meter))
                print(out_l.shape)
                sys.stdout.flush()         
    
    #COD 20/02/07 add temporal training for two images
    else:
        for idx, [(inputs1, _, index), (inputs2, _, lagged_index)] in enumerate(train_loader):
            data_time.update(time.time() - end)

            bsz = inputs1.size(0)
            inputs1 = inputs1.float()
            inputs2 = inputs2.float()

            if torch.cuda.is_available():
                index = index.cuda(non_blocking=True)
                inputs1 = inputs1.cuda()

                lagged_index = lagged_index.cuda(non_blocking=True)
                inputs2 = inputs2.cuda()

            # ===================forward=====================
            if not opt.pretrained:
                feat_one = model(inputs1)
                feat_two = model(inputs2)
            else:
                one_l, one_ab = model(inputs1)
                feat_one = one_l[:, np.new_axis]
                feat_one[:,:,2] = one_ab 
                print(feat_one.size())
                feat_two, _ = model(inputs2)
            out_one, out_two = contrast(feat_one, feat_two, index)

            one_loss = criterion_l(out_one) #l is naming convention only
            two_loss = criterion_ab(out_two) #ab is naming only
            one_prob = out_one[:, 0].mean()
            two_prob = out_two[:, 0].mean()

            loss = one_loss + two_loss

            # ===================backward=====================
            optimizer.zero_grad()
            if opt.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # ===================meters=====================
            losses.update(loss.item(), bsz)
            l_loss_meter.update(one_loss.item(), bsz)       #name only
            l_prob_meter.update(one_prob.item(), bsz)       #name only
            ab_loss_meter.update(two_loss.item(), bsz)      #name only
            ab_prob_meter.update(two_prob.item(), bsz)      #name only

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'one_p {oneprobs.val:.3f} ({oneprobs.avg:.3f})\t'
                    'two_p {twoprobs.val:.3f} ({twoprobs.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, oneprobs=l_prob_meter,
                    twoprobs=ab_prob_meter))
                print(out_one.shape)
                sys.stdout.flush()

    return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg   

def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_ab, criterion_l = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # set mixed precision
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        pre = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(pre['model'])
        del pre
    else:
        print('No pretrained found')

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        l_loss, l_prob, ab_loss, ab_prob = train(epoch, train_loader, model, contrast, criterion_l, criterion_ab,
                                                 optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger, set appropriate labels (COD 20/02/07)
        if not args.view == 'temporal':
            loss_label1 = 'l_loss'
            prob_label1 = 'l_prob'
            loss_label2 = 'ab_loss'
            prob_label2 = 'ab_prob'
        else:
            loss_label1 = 'img1_loss'
            prob_label1 = 'img1_prob'
            loss_label2 = 'img2_loss'
            prob_label2 = 'img2_prob'
        
        logger.log_value('{}'.format(loss_label1), l_loss, epoch)
        logger.log_value('{}'.format(prob_label1), l_prob, epoch)
        logger.log_value('{}'.format(loss_label2), ab_loss, epoch)
        logger.log_value('{}'.format(prob_label2), ab_prob, epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
