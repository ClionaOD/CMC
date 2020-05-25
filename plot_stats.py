import os
import pickle
import collections
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.spatial.distance as ssd
from scipy import stats
from skbio.stats.distance import mantel

act_path = './activations/main/'
binary_rdm = True
assoc_comp = False

chosenCategs = ['gown', 'hair', 'suit', 'coat', 'tie', 'shirt', 'sunglasses', 'shoe', 'screen', 'computer', 'table', 'food', 'restaurant', 'glass', 'alcohol', 'wine', 'lamp', 'couch', 'chair', 'closet', 'piano', 'pillow', 'desk', 'window', 'bannister']
clusters = {}
with open('./global_categs.pickle', 'rb') as f:
    categories = pickle.load(f)
for k, lst in categories.items():
    for label in lst:
        clusters[label] = k
clusters = {k:v for k,v in clusters.items() if k in chosenCategs}
cluster_df = pd.DataFrame.from_dict(clusters, orient='index')

#create binary model rdm - 0 indicates within category partner and 1 across
if binary_rdm:
    model_rdm = pd.DataFrame(data=np.ones((25,25)),index=chosenCategs, columns=chosenCategs)
    for k1, v1 in cluster_df.iterrows():
        for k2, v2 in cluster_df.iterrows():
            if v1[0] == v2[0]:
                model_rdm.loc[k1][k2] = 0
                model_rdm.loc[k2][k1] = 0
    np.fill_diagonal(model_rdm.values, 0)
elif assoc_comp:
    with open('/home/clionaodoherty/Desktop/associations/results/imagenet_categs.pickle','rb') as f:
        model_rdm = pickle.load(f)
    model_rdm = model_rdm['lag_1']
    model_rdm = -1 * (model_rdm + model_rdm.T)
    np.fill_diagonal(model_rdm.values, 0)
else:
    with open('./w2v_for_cmc.pickle','rb') as f:
        model_rdm = pickle.load(f)
    np.fill_diagonal(model_rdm.values, 0)

#choose the layer to evaluate
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']

stats_df = pd.DataFrame(index=[file.split('_')[0].split('.')[0] for file in os.listdir(act_path) if not os.path.isdir('{}{}'.format(act_path,file))], columns=layers)
sig_df = pd.DataFrame(index=[file.split('_')[0].split('.')[0] for file in os.listdir(act_path) if not os.path.isdir('{}{}'.format(act_path,file))], columns=layers)

for idx, x in enumerate(layers):
    chosenLayer = x
    for file in os.listdir(act_path):
        if os.path.isdir('{}{}'.format(act_path,file)):
            continue
        else:
            title = file.split('_')[0]
            title = title.split('.')[0]

            with open('{}/{}'.format(act_path,file),'rb') as f:
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
            
            #calculate rdm with euclidean distance
            rdm = ssd.pdist(df.values.T)
            rdm = ssd.squareform(rdm)
            rdm = pd.DataFrame(rdm, columns=list(layer_dict.keys()), index=list(layer_dict.keys()))

            #reindex rdm by category membership
            rdm = rdm.reindex(index=chosenCategs, columns=chosenCategs)   

            #statistical testing of category effect using kendall tau correlation and Mantel test
            corr, pval, n = mantel(rdm.values, model_rdm.values, method='kendalltau')
            stats_df.loc[title][chosenLayer] = corr
            if pval < (0.05 / 7):
                sig_df.loc[title][chosenLayer] = 1
            else:
                sig_df.loc[title][chosenLayer] = 0

stats_df = pd.DataFrame(stats_df.values, columns=stats_df.columns, index=['Lab ImageNet','random weights','1sec finetuned','60sec finetuned', 'Lab movie dataset', '30sec finetuned', '10sec finetuned'])
sig_df = pd.DataFrame(sig_df.values, columns=sig_df.columns, index=['Lab ImageNet','random weights','1sec finetuned','60sec finetuned', 'Lab movie dataset', '30sec finetuned', '10sec finetuned'])
stats_df = stats_df.reindex(['random weights', 'Lab ImageNet', 'Lab movie dataset' ,'1sec finetuned','10sec finetuned', '30sec finetuned', '60sec finetuned'])
sig_df = sig_df.reindex(stats_df.index)
fig, (ax1,leg) = plt.subplots(nrows=1,ncols=2,gridspec_kw={'width_ratios': [1,.3]})
fig.subplots_adjust(wspace=0.5)
sns.lineplot(data=stats_df.T.astype(float), ax=ax1, dashes=False)
handles, labels = ax1.get_legend_handles_labels()
ax1.get_legend().remove()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel('alexnet layer')
ax1.set_ylabel('coding of superordinate category')
leg.legend(handles, labels)
leg.axis('off') 

sigs=list(zip(np.where(sig_df==1)[0], np.where(sig_df==1)[1]))
for x in sigs:
    anot = (x[1] , stats_df.iloc[x[0]][x[1]])
    ax1.annotate('*', anot)

plt.savefig('/home/clionaodoherty/Desktop/cmc_figs/stats/stats_w_Lab_movie.pdf')
plt.show()