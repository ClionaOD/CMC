import os
import time
import collections
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from util import AverageMeter

from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from models.alexnet import TemporalAlexNetCMC
from dataset import ImageFolderInstance
from dataset import RGB2Lab

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--model_path', type=str, default=None, help='path of model to visualise')
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    opt = parser.parse_args()

    return opt

def set_model(args):
    model = TemporalAlexNetCMC()
    print('==> loading pre-trained model')
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
    print('==> done')

    model = model.cuda()
    model.eval()

    return model

def get_loader(args):
    folder = os.path.join(args.data_folder, 'val_in_folders')
    
    mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
    std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
    color_transfer = RGB2Lab()

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        color_transfer,
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(
        folder,
        transform=train_transform)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    return loader

def compute_features(loader, model, opt):
    print('Compute features')
    model.eval()
    act = {}

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        _model_feats.append(output.cpu().numpy())

    for m in model.features.modules():
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(_store_feats)

    for m in model.classifier.modules():
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(_store_feats)

    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)
            
            _model_feats = []
            aux = model(input).data.cpu().numpy()
            act[target[0]] = _model_feats
            print(i)

    return act

def get_activations(loader, model, opt):
    features = compute_features(loader, model, opt)
    return features

def main():
    args = parse_option()
    loader = get_loader(args)
    model = set_model(args)

if __name__ == "__main__":
    model_path = '/home/clionaodoherty/movie-associations/saves/temporal/finetune1sec/movie-training-1sec/pretrained_memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_1_view_temporal/ckpt_epoch_80.pth'
    model = set_model()
    loader = get_loader()

