from dataset import twoImageFolderInstance
from dataset import get_color_distortion
import torch
import math
import numpy as np
import pickle
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
    return h,rms

#data_folder = '/data/movie-associations/train/'
data_folder = '/home/clionaodoherty/Desktop/fyp2020/stimuli/'

train_transform = transforms.ToTensor()

rms_lags = {}

lags = list(np.linspace(0,60,25,dtype=int))
lags.extend(np.linspace(60,36000,15,dtype=int))

for lag in lags:
    print('==> calculating lag {}'.format(lag))
    
    train_dataset = twoImageFolderInstance(data_folder, time_lag=lag, transform=train_transform)
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=1000)
    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1,sampler=train_sampler)

    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    rms_dict = {}
    h_dict = {}

    for idx, [(inputs1, path1, index), (inputs2, path2, lagged_index)] in enumerate(train_loader):  
        categ = path1[0].split('/')[-2]
        if not categ in rms_dict.keys():
            rms_dict[categ] = []
        if not categ in h_dict.keys():
            h_dict[categ] = []

        im1 = transforms.ToPILImage()(torch.squeeze(inputs1))
        im2 = transforms.ToPILImage()(torch.squeeze(inputs2))

        h,rms = rmsdiff(im1,im2)
        rms_dict[categ].append(rms)
        h_dict[categ].append(h)

    rms_dict = {k:np.mean(v) for k,v in rms_dict.items()}

    rms_lags[str(lag)] = np.mean(list(rms_dict.values()))
            
df = pd.DataFrame.from_dict(rms_lags, orient='index')
#with open('/home/clionaodoherty/CMC/rms_1min.pickle','wb') as f:
#    pickle.dump(df,f)

#with open('./rms_1min.pickle','rb') as f:
#    rms=pickle.load(f)

fig, ax = plt.subplots()
ax.plot(df[:25])
ax.set_title('RMS*-1 of two images over increasing lags')
ax.set_xlabel('lag in seconds')
ax.set_ylabel('RMS difference')
plt.tight_layout()
plt.savefig('/home/clionaodoherty/Desktop/cmc_figs/img_rms.pdf')
plt.show()
