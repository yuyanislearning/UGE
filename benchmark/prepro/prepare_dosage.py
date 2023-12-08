import pandas as pd
from entrz_symbol import my_gene_get_gene_symbol


dosage = pd.read_csv('/local2/yuyan/gene_emb/data/benchmark/dosage/dosage_sens_tf_labels.csv')



sensitive = dosage["dosage_sensitive"].dropna()
insensitive = dosage["dosage_insensitive"].dropna()

sensitive = my_gene_get_gene_symbol(sensitive)
sensitive = [sensitive[i] for i in sensitive]
insensitive = my_gene_get_gene_symbol(insensitive)
insensitive = [insensitive[i] for i in insensitive]

# save
with open('/local2/yuyan/gene_emb/data/benchmark/dosage/dosage_sensitive.txt', 'w') as f:
    f.write('\n'.join(sensitive))

with open('/local2/yuyan/gene_emb/data/benchmark/dosage/dosage_insensitive.txt', 'w') as f:
    f.write('\n'.join(insensitive))

