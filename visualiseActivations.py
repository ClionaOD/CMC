import pickle
import os
import collections
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch

from scipy import stats
from scipy.spatial import procrustes
from sklearn.manifold import MDS
from skbio.stats.distance import mantel

def hierarchical_clustering(matrix, label_list, outpath=None):
    fig,ax = plt.subplots(figsize=(10,15))
    dend = sch.dendrogram(sch.linkage(matrix), 
        ax=ax, 
        labels=label_list, 
        orientation='left'
    )
    ax.tick_params(axis='x',labelrotation=35, labelsize=4)
    if outpath:
        plt.savefig(outpath)
    plt.close()

    cluster_order = dend['ivl']

    return cluster_order

act_path = './activations/'

#choose items and create dict of their clusters
chosenCategs = ['gown', 'hair', 'suit', 'coat', 'tie', 'shirt', 'sunglasses', 'shoe', 'screen', 'computer', 'table', 'food', 'restaurant', 'glass', 'alcohol', 'wine', 'lamp', 'couch', 'chair', 'closet', 'piano', 'pillow', 'desk', 'window', 'bannister']
clusters = {}
with open('./global_categs.pickle', 'rb') as f:
    categories = pickle.load(f)
for k, lst in categories.items():
    for label in lst:
        clusters[label] = k

#choose the layer to evaluate
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
chosenLayer = 'conv5'

#set RDM and MDS figures
fig1, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(
    nrows=2, 
    ncols=3,
    figsize=(11.69,8.27)
)
fig1.subplots_adjust(wspace=0.4, hspace=0.8)

fig2, ((leg,axs1,axs2),(b1,axs3,axs4),(b2,axs5,axs6)) = plt.subplots(
    nrows=3, 
    ncols=3, 
    figsize=(8.27,11.69), 
    gridspec_kw={'width_ratios': [0.1,5,5]}
)
b1.axis('off')
b2.axis('off')

ref = False
align = np.array([])

#plot activations
for file in os.listdir(act_path):
    if os.path.isdir('{}{}'.format(act_path,file)):
            continue
    else:
        title = file.split('_')[0]

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

        #do MDS on the euclidean distance matrix, set category for plot 
        mds = MDS(n_components=2, dissimilarity='precomputed')
        df_embedding = pd.DataFrame(mds.fit_transform(rdm.values), index=rdm.index)
        if ref == False:
            align = df_embedding.values
            ref = True
        else:
            mtx1, mtx2, disparity = procrustes(align, df_embedding.values)
            df_embedding = pd.DataFrame(mtx2, index=rdm.index)

        colour_column = [clusters[k] for k,v in df_embedding.iterrows()]
        df_embedding['category'] = colour_column

        #reindex rdm by category membership
        rdm = rdm.reindex(index=chosenCategs, columns=chosenCategs)

        #create binary model rdm - 0 indicates within category partner and 1 across
        model_rdm = pd.DataFrame(data=np.ones((25,25)),index=chosenCategs, columns=chosenCategs)
        for k1, v1 in df_embedding.iterrows():
            for k2, v2 in df_embedding.iterrows():
                if v1['category'] == v2['category']:
                    model_rdm.loc[k1][k2] = 0
                    model_rdm.loc[k2][k1] = 0
        np.fill_diagonal(model_rdm.values, 0)   

        #statistical testing of category effect using Pearson correlation and Mantel test
        corr, pval, n = mantel(rdm.values, model_rdm.values, method='kendalltau')
        print('{}: corr={} p={}'.format(title,corr,pval))

        #plot RDMs and MDS
        if 'random' in title:
            ax = sns.heatmap(rdm, ax=ax1)
            ax.set_title('Random weights', fontsize=10)
            ax.tick_params('y',labelsize=8)
            ax.tick_params('x', labelsize=7)

            sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['category'], ax=axs1)
            axs1.set_title('Random weights')
            axs1.set_xlabel(' ')
            axs1.set_ylabel(' ')
            axs1.axis('equal')
            axs1.spines['top'].set_visible(False)
            axs1.spines['right'].set_visible(False)
        elif 'lab' in title:
            ax = sns.heatmap(rdm,  ax=ax2)
            ax.set_title('Lab trained', fontsize=10)
            ax.tick_params('y',labelsize=8)
            ax.tick_params('x', labelsize=7)

            sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['category'], legend=False,  ax=axs2)
            axs2.set_title('Lab trained')
            axs2.set_xlabel(' ')
            axs2.set_ylabel(' ')
            axs2.axis('equal')
            axs2.spines['top'].set_visible(False)
            axs2.spines['right'].set_visible(False)
        elif '1sec' in title:
            ax = sns.heatmap(rdm,  ax=ax3)
            ax.set_title('Finetuned - 1 sec lag', fontsize=10)
            ax.tick_params('y',labelsize=8)
            ax.tick_params('x', labelsize=7)

            sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['category'], legend=False,  ax=axs3)
            axs3.set_title('1sec finetuned')
            axs3.set_xlabel(' ')
            axs3.set_ylabel(' ')
            axs3.axis('equal')
            axs3.spines['top'].set_visible(False)
            axs3.spines['right'].set_visible(False)
        elif '10sec' in title:
            ax = sns.heatmap(rdm,  ax=ax4)
            ax.set_title('Finetuned - 10 sec lag', fontsize=10)
            ax.tick_params('y',labelsize=8)
            ax.tick_params('x', labelsize=7)

            sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['category'], legend=False,  ax=axs4)
            axs4.set_title('10sec finetuned')
            axs4.set_xlabel(' ')
            axs4.set_ylabel(' ')
            axs4.axis('equal')
            axs4.spines['top'].set_visible(False)
            axs4.spines['right'].set_visible(False)
        elif '30sec' in title:
            ax = sns.heatmap(rdm,  ax=ax5)
            ax.set_title('Finetuned - 30 sec lag', fontsize=10)
            ax.tick_params('y',labelsize=8)
            ax.tick_params('x', labelsize=7)

            sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['category'], legend=False,  ax=axs5)
            axs5.set_title('30sec finetune')
            axs5.set_xlabel(' ')
            axs5.set_ylabel(' ')
            axs5.axis('equal')
            axs5.spines['top'].set_visible(False)
            axs5.spines['right'].set_visible(False)
        else:
            ax = sns.heatmap(rdm,  ax=ax6)
            ax.set_title('Finetuned - 60 sec lag', fontsize=10)
            ax.tick_params('y',labelsize=8)
            ax.tick_params('x', labelsize=7)

            sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['category'], legend=False,  ax=axs6)
            axs6.set_title('60sec finetuned')
            axs6.set_xlabel(' ')
            axs6.set_ylabel(' ')
            axs6.axis('equal')
            axs6.spines['top'].set_visible(False)
            axs6.spines['right'].set_visible(False)


handles, labels = axs1.get_legend_handles_labels()
axs1.get_legend().remove()
leg.legend(handles, labels)
leg.axis('off')

plt.tight_layout()
#fig1.savefig('/home/clionaodoherty/Desktop/cmc_figs/nice/rdms.pdf')
#fig2.savefig('/home/clionaodoherty/Desktop/cmc_figs/nice/mds.pdf')
plt.show()

#plt.close()
