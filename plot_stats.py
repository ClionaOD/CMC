import os
import pickle
import collections
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import scipy.spatial.distance as ssd
from scipy import stats
from skbio.stats.distance import mantel

chosenCategs = ['gown', 'hair', 'suit', 'coat', 'tie', 'shirt', 'sunglasses', 'shoe', 'screen', 'computer', 'table', 'food', 'restaurant', 'glass', 'alcohol', 'wine', 'lamp', 'couch', 'chair', 'closet', 'piano', 'pillow', 'desk', 'window', 'bannister']
clusters = {}
with open('./global_categs.pickle', 'rb') as f:
    categories = pickle.load(f)
for k, lst in categories.items():
    for label in lst:
        clusters[label] = k

#choose the layer to evaluate
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']

stats_df = pd.DataFrame(index=[file.split('_')[0] for file in os.listdir('./activations')], columns=layers)
sig_df = pd.DataFrame(index=[file.split('_')[0] for file in os.listdir('./activations')], columns=layers)
for idx, x in enumerate(layers):
    chosenLayer = x
    for file in os.listdir('./activations'):
        title = file.split('_')[0]

        with open('./activations/{}'.format(file),'rb') as f:
            activations = pickle.load(f)

        #limit to chosen categories & chosen layer
        activations = {k:activations[k] for k in chosenCategs}
        layer_dict = collections.OrderedDict((k,v[chosenLayer]) for k,v in activations.items())

        #construct df and mean cross dimensions if necessary
        df = pd.DataFrame()
        for k,v in layer_dict.items():
            if not layers.index(chosenLayer) > 4:
                s = pd.Series(np.mean(layer_dict[k], axis=(1,2)))
                df[k] = s
            else:
                s = pd.Series(layer_dict[k])
                df[k] = s

        #set categories
        clusters = {}
        with open('./global_categs.pickle', 'rb') as f:
            categories = pickle.load(f)
        for k, lst in categories.items():
            for label in lst:
                clusters[label] = k
        clusters = {k:v for k,v in clusters.items() if k in chosenCategs}
        cluster_df = pd.DataFrame.from_dict(clusters, orient='index')

        #calculate rdm with euclidean distance
        rdm = ssd.pdist(df.values.T)
        rdm = ssd.squareform(rdm)
        rdm = pd.DataFrame(rdm, columns=list(layer_dict.keys()), index=list(layer_dict.keys()))

        #reindex rdm by category membership
        rdm = rdm.reindex(index=chosenCategs, columns=chosenCategs)   

        #create binary model rdm - 0 indicates within category partner and 1 across
        model_rdm = pd.DataFrame(data=np.ones((25,25)),index=chosenCategs, columns=chosenCategs)
        for k1, v1 in cluster_df.iterrows():
            for k2, v2 in cluster_df.iterrows():
                if v1[0] == v2[0]:
                    model_rdm.loc[k1][k2] = 0
                    model_rdm.loc[k2][k1] = 0
        np.fill_diagonal(model_rdm.values, 0)

        #statistical testing of category effect using kendall tau correlation and Mantel test
        corr, pval, n = mantel(rdm.values, model_rdm.values, method='kendalltau')
        stats_df.loc[title][chosenLayer] = corr
        if pval < (0.05 / 7):
            sig_df.loc[title][chosenLayer] = 1
        else:
            sig_df.loc[title][chosenLayer] = 0

fig, (ax1,leg) = plt.subplots(nrows=1,ncols=2,gridspec_kw={'width_ratios': [1,.3]})
stats_df.T.plot.line(ax=ax1)
handles, labels = ax1.get_legend_handles_labels()
ax1.get_legend().remove()
leg.legend(handles, labels)
leg.axis('off')

sigs=list(zip(np.where(sig_df==1)[0], np.where(sig_df==1)[1]))
for x in sigs:
    anot = (x[1] , stats_df.iloc[x[0]][x[1]])
    ax1.annotate('*', anot)




print('wait')