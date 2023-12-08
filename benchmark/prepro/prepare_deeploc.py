import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/local2/yuyan/gene_emb/code/benchmark')
from entrz_symbol import my_gene_get_gene_symbol


train_dat = pd.read_csv('/local2/yuyan/gene_emb/data/benchmark/deeploc/Swissprot_Train_Validation_dataset.csv')
test_dat = pd.read_csv('/local2/yuyan/gene_emb/data/benchmark/deeploc/hpa_testset.csv')
id_map = pd.read_csv('/local2/yuyan/gene_emb/data/benchmark/deeploc/uid2symbol.tsv',sep='\t')

# filter to retain only human
id_map = id_map[id_map['Organism'].isin(['Homo sapiens (Human)'])]
id_map = id_map[~id_map['Gene Names (synonym)'].isnull()]
# dict from id to gene
id2gene = {uid:id_map['Gene Names (synonym)'].iloc[i].split(' ')[0] for i,uid in enumerate(id_map['From'])}
uids = list(id2gene.keys())

train_dat = train_dat[train_dat['ACC'].isin(uids)]
# convert id
train_dat['gene'] = train_dat['ACC'].map(id2gene)
train_dat = train_dat.drop(['Unnamed: 0', 'ACC','Kingdom', 'Partition', 'Sequence'], axis=1)

test_id_map = my_gene_get_gene_symbol(test_dat['sid'].values)
# found no hit

# combine duplicates, group by gene, sum other values up
train_dat = train_dat.groupby('gene').agg({col: 'sum' for col in train_dat.columns if col != 'gene'})
# convert value > 1 to 1
convert_to_1 = lambda x: 1 if x > 1 else x
train_dat[train_dat.columns.difference(['Category'])] = train_dat[train_dat.columns.difference(['Category'])].applymap(convert_to_1)



# save 
train_dat.to_csv('/local2/yuyan/gene_emb/data/benchmark/deeploc/deeploc.tsv', sep='\t')

