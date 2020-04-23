import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from scipy import stats
import pandas as pd
import collections

def hierarchical_clustering(matrix, label_list, outpath=None):
    fig,ax = plt.subplots(figsize=(10,10))
    dend = sch.dendrogram(sch.linkage(matrix, method='ward'), 
        ax=ax, 
        labels=label_list, 
        orientation='left'
    )
    ax.tick_params(axis='x', labelsize=4)
    if outpath:
        plt.savefig(outpath)
    plt.close()

    cluster_order = dend['ivl']

    return cluster_order

with open('./mean_activations.pickle','rb') as f:
    activations = pickle.load(f)

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
chosenLayer = 'conv5'

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
reorder = hierarchical_clustering(rdm_df.values, list(rdm_df.columns))
rdm_df = rdm_df.reindex(rdm_df, xticklabels=True, yticklabels=True)

sns.heatmap(rdm_df)
plt.show()
