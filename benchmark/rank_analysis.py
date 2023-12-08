'''
plot the ranking of different embeddings
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

in_path = '/local2/yuyan/gene_emb/data/benchmark/res/summary'

files = os.listdir(in_path)
files = [f for f in files if f.endswith('.tsv')]

# micro ranking
micro_ranking = pd.DataFrame()
for fle in files:
    dat = pd.read_csv(os.path.join(in_path, fle), sep='\t')
    # group by task, get ranking of auprc
    dat = dat.groupby('task').apply(lambda x: x.sort_values('auprc', ascending=False))
    dat['rank'] = dat.groupby('task').cumcount() + 1
    dat = dat.reset_index(drop=True)
    dat = dat[['task', 'emb','rank']]
    micro_ranking = pd.concat([micro_ranking, dat], axis=0)

# plot boxplot with scatter of ranking group by emb
sns.set_style('whitegrid')
sns.set_palette('Set2')
plt.figure(figsize=(8,6))
sns.boxplot(x='emb', y='rank', data=micro_ranking)
sns.stripplot(x='emb', y='rank', data=micro_ranking, color='black', size=4, jitter=True)
plt.xlabel('Embedding')
plt.ylabel('Ranking')
plt.title('Micro Ranking of Embeddings')
plt.savefig(os.path.join(in_path,'micro_ranking.png'), dpi=300)
plt.close()

# macro ranking
macro_ranking = pd.DataFrame()
for fle in files:
    dat = pd.read_csv(os.path.join(in_path, fle), sep='\t')
    # group by task, get ranking of auprc
    dat = dat.groupby('task').apply(lambda x: x.sort_values('auprc', ascending=False))
    dat['rank'] = dat.groupby('task').cumcount() + 1
    dat = dat.reset_index(drop=True)
    dat = dat[['emb','rank']]
    # get median rank by task
    dat = dat.groupby('emb').median().reset_index()

    macro_ranking = pd.concat([macro_ranking, dat], axis=0)


# plot boxplot with scatter of ranking group by emb
sns.set_style('whitegrid')
sns.set_palette('Set2')
plt.figure(figsize=(8,6))
sns.boxplot(x='emb', y='rank', data=macro_ranking)
sns.stripplot(x='emb', y='rank', data=macro_ranking, color='black', size=4, jitter=True)
plt.xlabel('Embedding')
plt.ylabel('Ranking')
plt.title('Macro Ranking of Embeddings')
plt.savefig(os.path.join(in_path,'macro_ranking.png'), dpi=300)
plt.close()



