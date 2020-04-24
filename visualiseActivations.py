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

fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(8.27,11.69))
fig.subplots_adjust(wspace=0.4, hspace=0.5)

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
        ax = sns.heatmap(rdm_df, ax=ax1)
        ax.set_title('Random Weights Network', pad=2)
        ax.tick_params('y',labelrotation=35)
    elif 'lab' in title:
        ax = sns.heatmap(rdm_df,  ax=ax2)
        ax.set_title('Lab Trained Network', pad=2)
        ax.tick_params('y',labelrotation=35)
    elif '1sec' in title:
        ax = sns.heatmap(rdm_df,  ax=ax3)
        ax.set_title('Finetuned - 1 sec Lag', pad=2)
        ax.tick_params('y',labelrotation=35)
    elif '10sec' in title:
        ax = sns.heatmap(rdm_df,  ax=ax4)
        ax.set_title('Finetuned - 10 sec Lag', pad=2)
        ax.tick_params('y',labelrotation=35)
    elif '30sec' in title:
        ax = sns.heatmap(rdm_df,  ax=ax5)
        ax.set_title('Finetuned - 30 sec Lag', pad=2)
        ax.tick_params('y',labelrotation=35)
    else:
        ax = sns.heatmap(rdm_df,  ax=ax6)
        ax.set_title('Finetuned - 60 sec Lag', pad=2)
        ax.tick_params('y',labelrotation=35)

plt.show()
