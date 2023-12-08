# Deep CCA
import sys
sys.path.append('../../DeepCCA')
from DeepCCAModels import DeepCCA
from linear_cca import linear_cca
from main import Solver
import numpy as np
import pdb
import torch
# import matplotlib.pyplot as plt


device = torch.device('cuda')

dat1_name = 'genept'
dat2_name = 'biolinkbert_summary'

input_shape1 = 1536
input_shape2 = 768
outdim_size = 10

layer_sizes1 = [128, 128, 128, outdim_size]
layer_sizes2 = [128, 128, 128, outdim_size]

learning_rate = 1e-3
batch_size = 128
n_epochs = 10
reg_par = 1e-5

use_all_singular_values = False

apply_linear_cca = True

out_dir = '/local2/yuyan/gene_emb/data/out'

def read_dat(data_type, n_dim=512):
    if data_type == 'gene2vec':
        gene2vec = np.load(f'{out_dir}/gene2vec.npy')
        gene2vec_gene = np.load(f'{out_dir}/gene2vec_gene.npy', allow_pickle=True)
        return gene2vec, gene2vec_gene
        
    if data_type == 'genept':
        genept = np.load(f'{out_dir}/genept.npy')
        genept_gene = np.load(f'{out_dir}/genept_gene.npy')
        return genept, genept_gene
        
    if data_type == 'biolinkbert_summary':
        biolinkbert_summary = np.load(f'{out_dir}/biolinkbert_summary.npy')
        biolinkbert_summary_gene = np.load(f'{out_dir}/biolinkbert_summary_gene.npy')
        return biolinkbert_summary, biolinkbert_summary_gene
        
    if data_type == 'random':
        gene_name = np.load(f'{out_dir}/genept_gene.npy')
        n = len(gene_name)
        random_emb = np.random.randn(n, n_dim)
        return random_emb, gene_name

def read_dat_2(dat1_name, dat2_name):
    dat1, dat1_gene = read_dat(dat1_name)
    dat2, dat2_gene = read_dat(dat2_name)

    gene_inter = set(dat1_gene).intersection(set(dat2_gene))

    print(f' {dat1_name} has {len(dat1_gene)} genes')
    print(f' {dat2_name} has {len(dat2_gene)} genes')
    print('intersection', len(gene_inter))

    # select the common genes
    dat1 = dat1[np.array([i for i,gene in enumerate(dat1_gene) if gene in gene_inter]), :]
    dat2 = dat2[np.array([i for i,gene in enumerate(dat2_gene) if gene in gene_inter]), :]

    # standardize the data
    dat1 = (dat1 - np.mean(dat1, axis=0)) / np.std(dat1, axis=0)
    dat2 = (dat2 - np.mean(dat2, axis=0)) / np.std(dat2, axis=0)

    return dat1, dat2

dat1, dat2 = read_dat_2(dat1_name, dat2_name)
model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1, input_shape2, outdim_size, use_all_singular_values, device=device)
l_cca = linear_cca()

solver = Solver(model, l_cca, outdim_size,n_epochs, batch_size, learning_rate, reg_par, device=device)
dat1 = torch.from_numpy(dat1)
dat2 = torch.from_numpy(dat2)

solver.fit(dat1, dat2)

loss, outputs = solver.test(dat1, dat2, apply_linear_cca)

pdb.set_trace()
# correlation = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
# print(correlation)

# # plot a scatter plot
# plt.scatter(X_c[:, 0], Y_c[:, 0], s=1)
# plt.xlabel('gene2vec')
# plt.ylabel('genept')
# plt.title('correlation: {}'.format(correlation))
# plt.savefig(f'{fig_dir}/gene2vec_genept_CCA.png')
# plt.show()