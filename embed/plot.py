from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from compare_all import dat_names

cor_matrix = np.load('/local2/yuyan/gene_emb/data/out/svCCA_cor_matrix.npy')
out_path = '/local2/yuyan/gene_emb/data/figs'

cor_matrix = cor_matrix + cor_matrix.T
# set diagonal to 1
np.fill_diagonal(cor_matrix, 1)

# Assuming similarity_matrix is your similarity matrix
distance_matrix = 1 - cor_matrix

from scipy.cluster.hierarchy import linkage, dendrogram

# Assuming distance_matrix is your distance matrix
linkage_matrix = linkage(distance_matrix, method='complete')
from scipy.cluster.hierarchy import leaves_list

row_order = leaves_list(linkage_matrix)
col_order = leaves_list(linkage_matrix)

reordered_similarity_matrix = cor_matrix[row_order][:, col_order]

sns.set(font_scale=0.8)  # Adjust font size as needed
ax = sns.heatmap(reordered_similarity_matrix, cmap='viridis', annot=True, fmt=".2f",
                 xticklabels=dat_names, yticklabels=dat_names)


plt.savefig(f'{out_path}/svCCA_cor_matrix_clustered.png', bbox_inches='tight')
plt.show()

