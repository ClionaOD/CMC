import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from scipy import stats
import pandas as pd
import collections

chosenCategs = ['coat', 'suit', 'prison', 'plant']
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
chosenLayer = 'conv5'

fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2)

for file in os.listdir('./activations'):
    with open('./activations/{}'.format(file),'rb') as f:
        activations = pickle.load(f)

    activations = {k:activations[k] for k in chosenCategs}

    layer_dict = collections.OrderedDict((k,list(v.values())[layers.index(chosenLayer)]) for k,v in activations.items())

    df = pd.DataFrame()
    for k,v in layer_dict.items():
        if not layers.index(chosenLayer) > 4:
            s = pd.Series(np.mean(layer_dict[k], axis=(1,2)))
            df[k] = s
        else:
            s = pd.Series(layer_dict[k])
            df[k] = s

    rdm = ssd.pdist(df.values.T)
    rdm = ssd.squareform(rdm)

    rdm_df = pd.DataFrame(rdm, columns=list(layer_dict.keys()), index=list(layer_dict.keys()))

    title = file.split('_')[0]
    if 'random' in title:
        ax = sns.heatmap(rdm_df, vmax=0.3, ax=ax1)
        ax.set_title(title)
    elif 'lab' in title:
        ax = sns.heatmap(rdm_df, vmax=0.3, ax=ax2)
        ax.set_title(title)
    elif '1sec' in title:
        ax = sns.heatmap(rdm_df, vmax=0.3, ax=ax3)
        ax.set_title(title)
    elif '10sec' in title:
        ax = sns.heatmap(rdm_df, vmax=0.3, ax=ax4)
        ax.set_title(title)
    elif '30sec' in title:
        ax = sns.heatmap(rdm_df, vmax=0.3, ax=ax5)
        ax.set_title(title)
    else:
        ax = sns.heatmap(rdm_df, vmax=0.3, ax=ax6)
        ax.set_title(title)

plt.show()
