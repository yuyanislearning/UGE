import pandas as pd
import numpy as np
import os
from entrz_symbol import batch_convert_ensembl_to_symbols
from tqdm import tqdm

file_dir = '/local2/yuyan/gene_emb/data/benchmark/dis_gene/'

dfs = []
for fle in os.listdir(file_dir):
    if fle.endswith('.tsv'):
        print(fle)
        df = pd.read_csv(file_dir + fle, sep='\t')
        df = df.drop(['pred', 'emb', 'fold'], axis=1)
        print('original df shape:', df.shape)
        # remove duplicate rows
        df = df.drop_duplicates()
        print('after drop duplicates:', df.shape)
        dfs.append(df)



df = pd.concat(dfs)
ensembl_ids = list(df['gene_id'].unique())
ensembl2symbol = {}
# batchly convert ensembl ids to gene symbols with batch size = 50
for i in tqdm(range(0, len(ensembl_ids), 50)):
    ensembl2symbol.update(batch_convert_ensembl_to_symbols(ensembl_ids[i:i+50]))



df['gene_symbol'] = df['gene_id'].map(ensembl2symbol)
df = df.drop('gene_id', axis=1)
df.to_csv(file_dir + 'dis_gene_final.txt', sep='\t', index=False)
