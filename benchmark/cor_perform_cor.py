'''
plot scatter plot of correlation between performance and correlation
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import seaborn as sns
sys.path.append('/local2/yuyan/gene_emb/code/embed')
from compare_all import dat_names

compare_cor_perform = False
dat_names = [dat_name.split('_')[0] for dat_name in dat_names]
cor_mat = np.load('/local2/yuyan/gene_emb/data/out/svCCA_cor_matrix.npy')
cor_mat = cor_mat+cor_mat.T+np.eye(cor_mat.shape[0])

in_path = '/local2/yuyan/gene_emb/data/benchmark/res/summary'

files = os.listdir(in_path)
files = [f for f in files if f.endswith('.tsv')]




if compare_cor_perform:
    # micro ranking
    micro_ranking = pd.DataFrame()
    for fle in files:
        print(fle)
        dat = pd.read_csv(os.path.join(in_path, fle), sep='\t')
        # group by task, substract the max of auprc column
        dat['diff_auprc'] = dat.groupby('task')['auprc'].transform(lambda x: x.max() - x)
        # group by task, get the emb column with the highest auprc 
        column_idx = dat.groupby('task')['auprc'].transform(lambda x: x.idxmax()).tolist()
        dat['best_emb'] = dat['emb'].iloc[column_idx].values

        # get the distance from best_emb to emb
        for i in range(dat.shape[0]):
            dat.loc[i, 'dist'] = 1 - cor_mat[dat_names.index(dat['emb'].iloc[i]), dat_names.index(dat['best_emb'].iloc[i])]

        # plot the distance vs diff_auprc and print the correlation on figure
        sns.set(style="whitegrid")
        ax = sns.scatterplot(x="dist", y="diff_auprc", data=dat)
        ax.set(xlabel='distance', ylabel='diff_auprc')
        ax.set_title(fle)
        
        correlation = np.corrcoef(dat['dist'], dat['diff_auprc'])[0, 1]
        ax.text(0.05, 0.95, 'correlation: '+str(correlation), transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
        plt.savefig(os.path.join(in_path,'cor_perform' ,fle.split('.')[0]+'.png'))
        plt.close()
