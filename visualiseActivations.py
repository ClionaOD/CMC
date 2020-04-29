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
from sklearn.manifold import MDS

def hierarchical_clustering(matrix, label_list, outpath=None):
    fig,ax = plt.subplots(figsize=(10,15))
    dend = sch.dendrogram(sch.linkage(matrix), 
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

#chosenCategs = ['coat', 'suit', 'prison', 'plant']
chosenCategs = ['gown', 'hair', 'suit', 'coat', 'tie', 'screen', 'table', 'food', 'restaurant', 'glass', 'alcohol', 'wine', 'lamp', 'couch', 'chair', 'closet', 'shirt', 'sunglasses', 'piano', 'window', 'bannister', 'pillow', 'desk', 'computer', 'shoe']
colours = {}
with open('./global_categs.pickle', 'rb') as f:
    categories = pickle.load(f)
for k, lst in categories.items():
    for label in lst:
        colours[label] = k

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
chosenLayer = 'conv5'

fig1, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(8.27,11.69))
fig1.subplots_adjust(wspace=0.4, hspace=0.5)

fig2, ((axs1,axs2),(axs3,axs4),(axs5,axs6)) = plt.subplots(nrows=3, ncols=2, figsize=(8.27,11.69))
fig2.subplots_adjust(wspace=0.4, hspace=0.5)

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

    # rdm = df.corr(method='kendall')
    # rdm = 1 - rdm

    rdm = pd.DataFrame(rdm, columns=list(layer_dict.keys()), index=list(layer_dict.keys()))

    mds = MDS(n_components=2, dissimilarity='precomputed')
    df_embedding = pd.DataFrame(mds.fit_transform(rdm), index=chosenCategs)

    colour_column = [colours[k] for k,v in df_embedding.iterrows()]
    df_embedding['cluster'] = colour_column

    order = list(df_embedding.sort_values(by='cluster').index)
    rdm = rdm.reindex(index=order, columns=order)

    model_rdm = pd.DataFrame(data=np.ones((25,25)),index=order, columns=order)
    for k1, v1 in df_embedding.iterrows():
        for k2, v2 in df_embedding.iterrows():
            if v1['cluster'] == v2['cluster']:
                model_rdm.loc[k1][k2] = 0
                model_rdm.loc[k2][k1] = 0
    np.fill_diagonal(model_rdm.values, 0)   

    tau, pval = stats.kendalltau(ssd.squareform(rdm.values), ssd.squareform(model_rdm.values))

    title = file.split('_')[0]
    print('{}: tau={} p={}'.format(title,tau,pval))
    #clusterOrder = hierarchical_clustering(ssd.squareform(rdm), chosenCategs, '/home/clionaodoherty/Desktop/cmc_figs/dends/euclidean/dend_{}'.format(title))
    #dm_df = rdm.reindex(index=clusterOrder, columns=clusterOrder)
    if 'random' in title:
        ax = sns.heatmap(rdm, ax=ax1)
        ax.set_title('Random Weights Network', pad=2)
        ax.tick_params('y',labelrotation=35)

        sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['cluster'], ax=axs1)
        axs1.set_title('Random')
        axs1.set_xlabel(' ')
        axs1.set_ylabel(' ')
        #for k,v in df_embedding.iterrows():
        #    axs1.annotate(k,v)
            #find a way of setting colour acc. to str value
    elif 'lab' in title:
        ax = sns.heatmap(rdm,  ax=ax2)
        ax.set_title('Lab Trained Network', pad=2)
        ax.tick_params('y',labelrotation=35)

        sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['cluster'], legend=False,  ax=axs2)
        axs2.set_title('Lab')
        axs2.set_xlabel(' ')
        axs2.set_ylabel(' ')
        #for k,v in df_embedding.iterrows():
        #    axs2.annotate(k,v)
    elif '1sec' in title:
        ax = sns.heatmap(rdm,  ax=ax3)
        ax.set_title('Finetuned - 1 sec Lag', pad=2)
        ax.tick_params('y',labelrotation=35)

        sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['cluster'], legend=False,  ax=axs3)
        axs3.set_title('1sec')
        axs3.set_xlabel(' ')
        axs3.set_ylabel(' ')
        #for k,v in df_embedding.iterrows():
        #    axs3.annotate(k,v)
    elif '10sec' in title:
        ax = sns.heatmap(rdm,  ax=ax4)
        ax.set_title('Finetuned - 10 sec Lag', pad=2)
        ax.tick_params('y',labelrotation=35)

        sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['cluster'], legend=False,  ax=axs4)
        axs4.set_title('10sec')
        axs4.set_xlabel(' ')
        axs4.set_ylabel(' ')
        #for k,v in df_embedding.iterrows():
        #    axs4.annotate(k,v)
    elif '30sec' in title:
        ax = sns.heatmap(rdm,  ax=ax5)
        ax.set_title('Finetuned - 30 sec Lag', pad=2)
        ax.tick_params('y',labelrotation=35)

        sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['cluster'], legend=False,  ax=axs5)
        axs5.set_title('30sec')
        axs5.set_xlabel(' ')
        axs5.set_ylabel(' ')
        #for k,v in df_embedding.iterrows():
        #    axs5.annotate(k,v)
    else:
        ax = sns.heatmap(rdm,  ax=ax6)
        ax.set_title('Finetuned - 60 sec Lag', pad=2)
        ax.tick_params('y',labelrotation=35)

        sns.scatterplot(x=df_embedding[0],y=df_embedding[1],hue=df_embedding['cluster'], legend=False,  ax=axs6)
        axs6.set_title('60sec')
        axs6.set_xlabel(' ')
        axs6.set_ylabel(' ')
        #for k,v in df_embedding.iterrows():
        #    axs6.annotate(k,v)

handles, labels = axs1.get_legend_handles_labels()
axs1.get_legend().remove()
fig2.legend(handles, labels, loc='upper left')
plt.show()
