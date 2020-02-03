import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import ImageFolderInstance

dataPath='/movie-associations'
data_folder = os.path.join(dataPath, 'train')
train_dataset = ImageFolderInstance(data_folder, transform=transforms.ToTensor())

loader = DataLoader(
    train_dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
)

mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print('Mean: {}'.format(mean))
print('Std: {}'.format(std))

