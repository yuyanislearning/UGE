import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from entrz_symbol import convert_ensp_to_gene_symbol


ppi_path = '/local2/yuyan/gene_emb/data/benchmark/string/9606.protein.physical.links.v12.0.txt'
ppi = pd.read_csv(ppi_path, sep=' ')

ppi['protein1'] = ppi['protein1'].apply(lambda x: x.split('.')[1])
ppi['protein2'] = ppi['protein2'].apply(lambda x: x.split('.')[1])

# convert ensembl to symbol
ensembl_ids = ppi['protein1'].tolist() + ppi['protein2'].tolist()
ensembl_ids = list(set(ensembl_ids))

# write to file
with open('/local2/yuyan/gene_emb/data/benchmark/ppi_ensembl_ids.txt', 'w') as f:
    for item in ensembl_ids:
        f.write("%s\n" % item)

ensg2symbol = pd.read_csv('/local2/yuyan/gene_emb/data/benchmark/string/ensg2symbol.tsv', sep='\t')
# retrieve first gene name in Gene names column
ensg2symbol['Gene Names'] = ensg2symbol['Gene Names'].astype(str)
ensg2symbol['Gene Names'] = ensg2symbol['Gene Names'].apply(lambda x: x.split(' ')[0])
# remove duplicated From and To
ensg2symbol = ensg2symbol.drop_duplicates(subset=['From'])

# convert ensembl to symbol
ensembl2symbol = {}
for i in tqdm(range(ensg2symbol.shape[0])):
    ensembl2symbol[ensg2symbol.iloc[i, 0]] = ensg2symbol.iloc[i, 3]

# filter on ppi, remove proteins without symbol
ppi = ppi[ppi['protein1'].isin(ensembl2symbol.keys())]
ppi = ppi[ppi['protein2'].isin(ensembl2symbol.keys())]

ppi['protein1'] = ppi['protein1'].apply(lambda x: ensembl2symbol[x])
ppi['protein2'] = ppi['protein2'].apply(lambda x: ensembl2symbol[x])

ppi['combined_score'] = 1

# random sample protein pairs to be negative samples
n = ppi.shape[0]
neg_ppi = pd.DataFrame.from_dict({'protein1':np.repeat(0, n), 'protein2':np.repeat(0, n), 'combined_score':np.repeat(0, n)})
symbol_ids = list(set(ppi['protein1'].tolist() + ppi['protein2'].tolist()))

def sample_neg_pair(n, ppi):
    p1 = np.random.choice(symbol_ids, size=n)
    p2 = np.random.choice(symbol_ids, size=n)
    
    # remove p1==p2
    ind = p1 != p2
    p1 = p1[ind]
    p2 = p2[ind]
    
    # remove existing pairs
    existing_pairs = set(zip(ppi['protein1'], ppi['protein2']))
    ind = np.array([(p1[i], p2[i]) not in existing_pairs and (p2[i], p1[i]) not in existing_pairs for i in range(len(p1))])
    p1 = p1[ind]
    p2 = p2[ind]
    
    return pd.DataFrame.from_dict({'protein1':p1, 'protein2':p2, 'combined_score':np.repeat(0, len(p1))})


neg_ppi = sample_neg_pair(n, ppi)
n = n - neg_ppi.shape[0]
print(n)
while n > 0:
    new_neg_ppi = sample_neg_pair(n, ppi)

    neg_ppi = pd.concat([neg_ppi, new_neg_ppi], axis=0)
    n = n - new_neg_ppi.shape[0]
    print(n)
    

# concat pos and neg
ppi = pd.concat([ppi, neg_ppi], axis=0)
ppi = ppi.sample(frac=1).reset_index(drop=True)

# save
ppi.to_csv('/local2/yuyan/gene_emb/data/benchmark/ppi.tsv',sep='\t', index=False)

