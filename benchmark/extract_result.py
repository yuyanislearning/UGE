'''
retrieve the result from the saved tsv files
for each task, each embedding, only retrieve the best result
format it into a dataframe
'''
import pandas as pd
import os
import numpy as np


in_path = '/local2/yuyan/gene_emb/data/benchmark/res/dis_gene'
out_path = '/local2/yuyan/gene_emb/data/benchmark/res/summary'
# get all file ends with .tsv
files = [f for f in os.listdir(in_path) if f.endswith('.tsv')]
all_df = pd.DataFrame()

for fle in files:
    # get the task name
    task = fle.split('_')[2]
    emb = fle.split('_')[3].split('.')[0]
    # read the tsv file
    df = pd.read_csv(os.path.join(in_path, fle), sep='\t')
    # get the best result
    df = df.sort_values(by=['auprc'], ascending=False)
    df = df.iloc[:1, :]
    df['task'] = task
    df['emb'] = emb
    # append to the dataframe
    all_df = all_df.append(df, ignore_index=True)
    
# save the dataframe
all_df = all_df.sort_values(by=['auprc'], ascending=False)
all_df.to_csv(os.path.join(out_path, 'all_dis_gene.tsv'), sep='\t', index=False)

