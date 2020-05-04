from dataset import twoImageFolderInstance
from dataset import get_color_distortion
import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageChops
from torchvision import transforms, datasets
from dataset import RGB2Lab
from models.alexnet import TemporalAlexNetCMC
from models.LinearModel import LinearClassifierAlexNet

def rmsdiff(im1, im2):
    diff = ImageChops.difference(im1,im2)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
    return rms

data_folder = '/data/movie-associations/train/'

train_transform = transforms.ToTensor()

rms_lags = {}

for lag in range(0,65,5):
    print('==> calculating lag {}'.format(lag))
    
    train_dataset = twoImageFolderInstance(data_folder, time_lag=lag, transform=train_transform)

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1)

    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    rms_dict = {}

    for idx, [(inputs1, path1, index), (inputs2, path2, lagged_index)] in enumerate(train_loader):
        print(path1)
        while index < 1000:    
            categ = path1[0].split('/')[-2]
            if not categ in rms_dict:
                rms_dict[categ] = []

            im1 = transforms.ToPILImage()(torch.squeeze(inputs1))
            im2 = transforms.ToPILImage()(torch.squeeze(inputs2))

            rms = rmsdiff(im1,im2)
            rms_dict[categ].append(rms)

    rms_dict = {k:np.mean(v) for k,v in rms_dict.items()}

    rms_lags[str(lag)] = np.mean(list(rms_dict.values()))

df = pd.DataFrame.from_dict(rms_lags, orient='index')
plt.plot(df)
plt.show()