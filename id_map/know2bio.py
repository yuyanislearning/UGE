import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../benchmark/')
from entrz_symbol import get_gene_info

id_map = '/home/arpelletier/workspace/know2bio/2023-11-20_know2bio/Know2BIO/benchmark/data/whole/entity2id.txt'
embed = '/home/arpelletier/workspace/know2bio/2023-11-20_know2bio/Know2BIO/benchmark/logs/11_29/whole/embedding_get_emb/embeddings.npy'

id_map = pd.read_csv(id_map, sep='\t', header=None, skiprows=1)
embed = np.load(embed)

# retain Entrez id only
id_map = id_map[id_map[0].str.contains('Entrez:')]
id_map[0] = id_map[0].str.replace('Entrez:', '')

# convert to gene symbol
entrez_id_list = id_map[0].tolist()
gene_info = {entrez_id:get_gene_info(entrez_id) for entrez_id in entrez_id_list}



# conver entrez id to gene symbol

print(id_map.shape)
