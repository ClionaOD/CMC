"""
Train CMC on temporal objective with AlexNet or ResNet50
This is mostly the same code as train_CMC.py
"""
from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket

import tensorboard_logger as tb_logger

from optim.lars import LARS                                 #Use You et al. '17 LARS optimizer for large batch training
from warmup_scheduler import GradualWarmupScheduler         #Linear warmup for 10 epochs then cosine decay schedule

from torchvision import transforms, datasets
from dataset import get_color_distortion
from util import AverageMeter, adjust_learning_rate

from models.alexnet import TemporalAlexNetCMC 
from models.resnet import TemporalResnetCMC
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from dataset import twoImageFolderInstance 

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
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')                   #Use bsz based on Chen et al. 2020 Table C.1
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.3, help='learning rate')          #lr = 0.3 * bsz/256 
    parser.add_argument('--lr_decay_epochs', type=str, default='300,340,360', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=10e-6, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet',
                                                                         'resnet50'])
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
    parser.add_argument('--view', type=str, default='temporal', choices=['temporal'])

    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    # range for timelag to test (COD 20/02/06)
    parser.add_argument('--time_lag', type=int, default=60, help='number of 1 second frames to lag by')

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

    #Use strong color distortion from Chen et al. Same mean and std as Lab
    if args.view == 'temporal':                                                      
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = get_color_distortion()
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

    #Load two images for temporal contrast
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
        model = TemporalAlexNetCMC(args.feat_dim)
    elif args.model.startswith('resnet'):
        model = TemporalResnetCMC()
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax) 
    criterion_one = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_two = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_two = criterion_two.cuda()
        criterion_one = criterion_one.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion_two, criterion_one

def set_optimizer(args, model):
    # return optimizer
    optimizer = LARS(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)

    return optimizer

def train(epoch, train_loader, model, contrast, criterion_one, criterion_two, optimizer, opt):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    one_loss_meter = AverageMeter()
    two_loss_meter = AverageMeter()
    one_prob_meter = AverageMeter()
    two_prob_meter = AverageMeter()

    end = time.time()

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
        feat_one = model(inputs1)
        feat_two = model(inputs2)
        out_one, out_two = contrast(feat_one, feat_two, index)

        one_loss = criterion_one(out_one) 
        two_loss = criterion_two(out_two) 
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
        one_loss_meter.update(one_loss.item(), bsz)      
        one_prob_meter.update(one_prob.item(), bsz)      
        two_loss_meter.update(two_loss.item(), bsz)      
        two_prob_meter.update(two_prob.item(), bsz)      

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
                data_time=data_time, loss=losses, oneprobs=one_prob_meter,
                twoprobs=two_prob_meter))
            print(out_one.shape)
            sys.stdout.flush()

    return one_loss_meter.avg, one_prob_meter.avg, two_loss_meter.avg, two_prob_meter.avg 

def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_two, criterion_one = set_model(args, n_data)

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

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler_cosine)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        #if epoch >= 300:
        #    adjust_learning_rate(epoch, args, optimizer)
        #else:
        scheduler_warmup.step()
        print("==> training...")

        time1 = time.time()
        one_loss, one_prob, two_loss, two_prob = train(epoch, train_loader, model, contrast, criterion_one, criterion_two,
                                                 optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        loss_label1 = 'img1_loss'
        prob_label1 = 'img1_prob'
        loss_label2 = 'img2_loss'
        prob_label2 = 'img2_prob'
        
        logger.log_value('{}'.format(loss_label1), one_loss, epoch)
        logger.log_value('{}'.format(prob_label1), one_prob, epoch)
        logger.log_value('{}'.format(loss_label2), two_loss, epoch)
        logger.log_value('{}'.format(prob_label2), two_prob, epoch)

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