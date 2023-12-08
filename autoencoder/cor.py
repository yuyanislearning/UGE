import sys
sys.path.append('../embed')
from compare_all import dat_names, read_dat, fit_svcca_model
import numpy as np



suffix = 'all_2023-12-01_15-02-51'
dat = f'/local2/yuyan/gene_emb/data/autoencoder/{suffix}.npy'
dat_gene = f'/local2/yuyan/gene_emb/data/autoencoder/{suffix}_gene.npy'
dat = np.load(dat)
dat_gene = np.load(dat_gene)

cors = []
for dat_i in range(len(dat_names)):
    dat2_name = dat_names[dat_i]
    dat2, dat2_gene = read_dat(dat2_name)
    
    gene_inter = set(dat_gene).intersection(set(dat2_gene))

    dat = dat[np.array([i for i,gene in enumerate(dat_gene) if gene in gene_inter]), :]
    dat2 = dat2[np.array([i for i,gene in enumerate(dat2_gene) if gene in gene_inter]), :]

    # sort the data based on genes
    dat_gene = [gene for gene in dat_gene if gene in gene_inter]
    dat2_gene = [gene for gene in dat2_gene if gene in gene_inter]

    dat = dat[np.argsort(dat_gene), :]
    dat2 = dat2[np.argsort(dat2_gene), :]

    # standardize the data
    dat = (dat - np.mean(dat, axis=0)) / np.std(dat, axis=0)
    dat2 = (dat2 - np.mean(dat2, axis=0)) / np.std(dat2, axis=0)

    svcca_cor, adj_svcca_cor, p_value = fit_svcca_model(dat, dat2, permute=False)
    cors.append(svcca_cor)

print(cors)
print(np.mean(cors))


