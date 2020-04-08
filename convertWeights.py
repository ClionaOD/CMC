import torch
import numpy as np
import collections

from models.alexnet import TemporalAlexNetCMC

pretrained = torch.load('/home/clionaodoherty/CMC/models/CMC_alexnet.pth')
pretrained = pretrained['model']
l_to_ab = collections.OrderedDict((k,v) for k,v in pretrained.items() if 'l_to_ab' in k)
ab_to_l = collections.OrderedDict((k,v) for k,v in pretrained.items() if 'ab_to_l' in k)

temporal = torch.load('/home/clionaodoherty/movie-associations/saves/temporal/movie-pretrain-1min/memory_nce_16384_alexnet_lr_0.6_decay_1e-05_bsz_512_sec_60_view_temporal/ckpt_epoch_1.pth')
temporal = collections.OrderedDict((k,v) for k,v in temporal.items() if k == 'model' or k =='epoch')
temp_layers = list(temporal['model'].keys())
temp_weights = list(temporal['model'].values())

l_weights = list(l_to_ab.values())
ab_weights = list(ab_to_l.values())

for idx, w in enumerate(l_weights):
    s = temp_weights[idx].size()
    if len(s) == 4:
        if idx == 0:
            j = 1
        else:
            j = s[1] // 2
        temp_weights[idx][:s[0]//2, :j , :, :] = w
    elif len(s) == 2 and not s[0] == 128:
        temp_weights[idx][:s[0]//2, :s[1]//2] = w
    elif len(s) == 2 and s[0] == 128:
        temp_weights[idx][:, :s[1]//2] = w
    elif len(s) == 0:
        pass
    else:
        if not idx == 45:
            temp_weights[idx][:s[0]//2] = w
        else:
            temp_weights[idx][:] = torch.rand(128)

for idx, w in enumerate(ab_weights):
    s = temp_weights[idx].size()
    if len(s) == 4:
        if idx == 0:
            j = 1
        else:
            j = s[1] // 2
        temp_weights[idx][s[0]//2:, j: , :, :] = w
    elif len(s) == 2 and not s[0] == 128:
        temp_weights[idx][s[0]//2:, s[1]//2:] = w
    elif len(s) == 2 and s[0] == 128:
        temp_weights[idx][:, s[1]//2:] = w
    elif len(s) == 0:
        pass
    else:
        if not idx == 45:
            temp_weights[idx][s[0]//2:] = w
        else:
            temp_weights[idx][:] = torch.rand(128)

new_layers = collections.OrderedDict(list(zip(temp_layers,temp_weights)))
temporal['model'] = new_layers

model = TemporalAlexNetCMC()
model.load_state_dict(temporal['model'])
torch.save(model.state_dict(), '/home/clionaodoherty/movie-associations/saves/Lab_pretrained_fullAlexNet.pth')
    



