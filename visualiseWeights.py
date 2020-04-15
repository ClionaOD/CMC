import torch
import collections
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns

model = torch.load('/home/clionaodoherty/movie-associations/saves/temporal/finetune1sec/movie-training-1sec/pretrained_memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_1_view_temporal/ckpt_epoch_80.pth')
weights = collections.OrderedDict([(k,v) for k,v in model['model'].items() if 'weight' in k])

fc7_conv = weights['encoder.module.alexnet.fc7.0.weight'].cpu()
rdm = distance.squareform(distance.pdist(fc7_conv, metric='correlation'))

fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(rdm, ax=ax, cmap='Blues_r')
plt.show()
plt.close()


