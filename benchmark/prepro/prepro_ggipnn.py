import pandas as pd

in_path = '/local2/yuyan/gene_emb/data/benchmark/GGIPNN'

ys= []
for tvt in ['train', 'valid', 'test']:
    df = pd.read_csv(f'{in_path}/{tvt}_label.txt', sep=' ', header=None)
    ys.extend(df[0].tolist())

gene_paris = []
for tvt in ['train', 'valid', 'test']:
    df = pd.read_csv(f'{in_path}/{tvt}_text.txt', sep=' ', header=None)
    gene_pair = zip(df[0].tolist(), df[1].tolist())
    gene_paris.extend(gene_pair)

print(len(ys), len(gene_paris))

# save
with open(f'{in_path}/ggipnn_gene_pairs.txt', 'w') as f:
    for gene_pair in gene_paris:
        f.write(f'{gene_pair[0]}\t{gene_pair[1]}\n')

with open(f'{in_path}/ggipnn_labels.txt', 'w') as f:
    for y in ys:
        f.write(f'{y}\n')

    
    
