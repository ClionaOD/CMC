import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from models.alexnet import TemporalAlexNetCMC
from dataset import RGB2Lab
from dataset import ImageFolderWithPaths

#freq items
with open('./freq_order.pickle','rb') as f:
    order = pickle.load(f)

with open('./synset_df.pickle','rb') as f:
    synsets = pickle.load(f)

#Load the pretrained model and set it to eval mode
model_path = '/home/clionaodoherty/movie-associations/saves/temporal/finetune1sec/movie-training-1sec/pretrained_memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_1_view_temporal/ckpt_epoch_80.pth'
model = TemporalAlexNetCMC()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['model'])
model.cuda()
model.eval()

#Define the transform and dataloader
folder = '/home/clionaodoherty/imagenet_samples/'
    
mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
color_transfer = RGB2Lab()

normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    color_transfer,
    transforms.ToTensor(),
    normalize,
])

#This will return (sample, class_index) as transformed by above
dataset = ImageFolderWithPaths(
    folder,
    transform=train_transform)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False)

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

for idx, (input, target, path) in enumerate(loader):
    model.encoder.module.alexnet.fc7.register_forward_hook(get_activation(path[0].split('/')[-2])) 
    input = input.float()
    input = input.cuda()
    feat = model(input)

activation = {k[0].split('/')[-2]:v for k,v in activation.items()}


        

