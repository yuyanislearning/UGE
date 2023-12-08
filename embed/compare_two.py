from sklearn.cross_decomposition import CCA
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from CKA import linear_CKA
import pdb
import os
from tqdm import tqdm
import cca_core
from compare_all import process_dat, read_dat, fit_svcca_model, fit_cka_model

fit_cka = True
fit_svcca = False

fig_dir = '/local2/yuyan/gene_emb/data/figs'
out_dir = '/local2/yuyan/gene_emb/data/out'

dat1_name = 'gene2vec' # gene2vec
dat1, dat1_gene = read_dat(dat1_name)

dat2_name = 'test'
dat2 = np.load(f'/local2/yuyan/gene_emb/data/autoencoder/{dat2_name}.npy')
dat2_gene = np.load(f'/local2/yuyan/gene_emb/data/autoencoder/{dat2_name}_gene.npy')

gene_inter = set(dat1_gene).intersection(set(dat2_gene))
# select the common genes
dat1 = dat1[np.array([i for i,gene in enumerate(dat1_gene) if gene in gene_inter]), :]
dat2 = dat2[np.array([i for i,gene in enumerate(dat2_gene) if gene in gene_inter]), :]
# sort the data based on genes
dat1_gene = [gene for gene in dat1_gene if gene in gene_inter]
dat2_gene = [gene for gene in dat2_gene if gene in gene_inter]

dat1 = dat1[np.argsort(dat1_gene), :]
dat2 = dat2[np.argsort(dat2_gene), :]
# standardize the data
dat1 = (dat1 - np.mean(dat1, axis=0)) / np.std(dat1, axis=0)
dat2 = (dat2 - np.mean(dat2, axis=0)) / np.std(dat2, axis=0)

# fit the model
if fit_svcca:
    svcca_cor = fit_svcca_model(dat1, dat2)
    # cka_cor = fit_cka_model(dat1, dat2)
    print(svcca_cor)
if fit_cka:
    cka_cor = fit_cka_model(dat1, dat2)
    print(cka_cor)
