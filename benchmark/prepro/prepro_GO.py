import pandas as pd
import numpy as np
from goatools import obo_parser
from go_util import trace_to_root_level, BP_childs
from pprint import pprint
from tqdm import tqdm

go_path = '/local2/yuyan/gene_emb/data/benchmark/GO/goa_human.gaf'
goa = pd.read_csv(go_path, sep='\t', header=None, comment='!', usecols=[2, 3, 4, 6], names=['gene','quali', 'go', 'evidence'])

print('before filter: ')
print(goa.shape)

# filter out  quali contains 'NOT'
goa = goa[~goa['quali'].str.contains('NOT')]

# retain experimental evidence
exp_evidence = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP','TAS', 'IC']
goa = goa[goa['evidence'].isin(exp_evidence)]

# trace back to one level below root
go = obo_parser.GODag('/local2/yuyan/gene_emb/data/benchmark/GO/go-basic.obo')
go_dict = {gene:[] for gene in goa['gene'].unique()}
print('after filter: ')
print(goa.shape)

for gene in tqdm(go_dict):
    trace = [trace_to_root_level( go_id, go) for go_id in goa[goa['gene']==gene]['go'].values]
    trace = list(set(trace))
    trace = [go_id for go_id in trace if go_id in BP_childs]
    go_dict[gene] = trace

# check distribution of go terms
BP_childs_count = {go_id:0 for go_id in BP_childs}
for gene in go_dict:
    for go_id in go_dict[gene]:
        BP_childs_count[go_id] += 1



# keep GO terms with at least 1000 genes
BP_childs_count = {go_id:count for go_id, count in BP_childs_count.items() if count >= 1000}
print('after filter: ')
pprint(BP_childs_count)

# filter out GO term not in BP_childs_count
for gene in go_dict:
    go_dict[gene] = [go_id for go_id in go_dict[gene] if go_id in BP_childs_count]

# filter out genes with no GO terms
go_dict = {gene:go_ids for gene, go_ids in go_dict.items() if len(go_ids)>0}
print('after filter: ')
print(len(go_dict))

# construct dataframe, where each row is a gene, and each column is a GO term

go_df = pd.DataFrame(index=go_dict.keys(), columns=BP_childs_count.keys())
go_df = go_df.fillna(0)
for gene in tqdm(go_dict):
    for go_id in go_dict[gene]:
        go_df.loc[gene, go_id] = 1
    
go_df.index.name = 'gene'    

# save to file
go_df.to_csv('/local2/yuyan/gene_emb/data/benchmark/GO/go_BP.tsv', sep='\t')




