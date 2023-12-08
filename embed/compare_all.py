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
import multiprocessing
import sys
sys.path.append('/local2/yuyan/gene_emb/code/benchmark/')
from entrz_symbol import batch_convert_ensembl_to_symbols, convert_ensp_to_gene_symbol


fig_dir = '/local2/yuyan/gene_emb/data/figs'
out_dir = '/local2/yuyan/gene_emb/data/out'

fit_cca = False
fit_cka = False
fit_svcca = True

PERMUTE = False

n_sample = 100 # 
dat_names = ['autoencoder','biogpt','string_stru2vec','omics','prottrans','geneformer','gene2vec', 'genept', 'biolinkbert_summary']
num_processes = 20


def main():

    # remove duplicates
    # for dat_name in dat_names:
    #     dat, dat_gene = read_dat(dat_name)
    #     dat, dat_gene = remove_dup(dat, dat_gene)
    #     np.save(f'{out_dir}/{dat_name}.npy', dat)
    #     np.save(f'{out_dir}/{dat_name}_gene.npy', dat_gene)


    CCA_cor_matrix = np.zeros((len(dat_names), len(dat_names)))
    CCA_cor_adj_matrix = np.zeros((len(dat_names), len(dat_names)))
    CKA_cor_matrix = np.zeros((len(dat_names), len(dat_names)))
    CKA_cor_adj_matrix = np.zeros((len(dat_names), len(dat_names)))
    svCCA_cor_matrix = np.zeros((len(dat_names), len(dat_names)))
    svCCA_cor_adj_matrix = np.zeros((len(dat_names), len(dat_names)))
    svCCA_p_value_matrix = np.zeros((len(dat_names), len(dat_names)))
    CKA_p_value_matrix = np.zeros((len(dat_names), len(dat_names)))

    for dat_i in range(len(dat_names)-1):
        for dat_j in tqdm(range(dat_i+1, len(dat_names))):
            dat1, dat2, dat1_name, dat2_name = process_dat(dat_i, dat_j, dat_names)
            # CCA
            if fit_cca:
                actual_correlation, adjust_correlation = fit_cca_model(dat1, dat2, dat1_name, dat2_name)
                # print(f'{dat1_name} {dat2_name} CCA correlation: ', actual_correlation)
                CCA_cor_matrix[dat_i, dat_j] = actual_correlation
                CCA_cor_adj_matrix[dat_i, dat_j] = adjust_correlation
                
            ## svCCA
            if fit_svcca:
                svcca_cor, adj_svcca_cor, p_value = fit_svcca_model(dat1, dat2, permute=PERMUTE)
                # print('svCCA correlation: ', svcca_cor)
                svCCA_cor_matrix[dat_i, dat_j] = svcca_cor
                svCCA_cor_adj_matrix[dat_i, dat_j] = adj_svcca_cor
                svCCA_p_value_matrix[dat_i, dat_j] = p_value
                # print('adjusted svCCA correlation: ', adj_svcca_cor)

            ## CKA
            if fit_cka:
                cka_cor, adj_cka_cor, p_value = fit_cka_model(dat1, dat2, permute=PERMUTE)
                # print('CKA correlation: ', cka_cor)
                CKA_cor_matrix[dat_i, dat_j] = cka_cor
                CKA_cor_adj_matrix[dat_i, dat_j] = adj_cka_cor
                CKA_p_value_matrix[dat_i, dat_j] = p_value

         

    if fit_cca:
        plot_cor_matrix(CCA_cor_matrix, dat_names, 'CCA')
        np.save(f'{out_dir}/CCA_cor_matrix.npy', CCA_cor_matrix)
        if perm_test_cca:
            plot_cor_matrix(CCA_cor_adj_matrix, dat_names, 'CCA_adj')
            np.save(f'{out_dir}/CCA_cor_adj_matrix.npy', CCA_cor_adj_matrix)
    
    if fit_svcca:

        plot_cor_matrix(svCCA_cor_matrix, dat_names, 'svCCA')
        np.save(f'{out_dir}/svCCA_cor_matrix.npy', svCCA_cor_matrix)
        if PERMUTE:
            plot_cor_matrix(svCCA_cor_adj_matrix, dat_names, 'svCCA_adj')
            np.save(f'{out_dir}/svCCA_cor_adj_matrix.npy', svCCA_cor_adj_matrix)
            plot_cor_matrix(svCCA_p_value_matrix, dat_names, 'svCCA_p_value')
            np.save(f'{out_dir}/svCCA_p_value_matrix.npy', svCCA_p_value_matrix)



    if fit_cka:
        plot_cor_matrix(CKA_cor_matrix, dat_names, 'CKA')
        np.save(f'{out_dir}/CKA_cor_matrix.npy', CKA_cor_matrix)
        plot_cor_matrix(CKA_cor_matrix, dat_names, 'CKA')
        np.save(f'{out_dir}/CKA_cor_matrix.npy', CKA_cor_matrix)
        if PERMUTE:
            plot_cor_matrix(CKA_cor_adj_matrix, dat_names, 'CKA_adj')
            np.save(f'{out_dir}/CKA_cor_adj_matrix.npy', CKA_cor_adj_matrix)
            plot_cor_matrix(CKA_p_value_matrix, dat_names, 'CKA_p_value')
            np.save(f'{out_dir}/CKA_p_value_matrix.npy', CKA_p_value_matrix)


def fit_cka_model(dat1, dat2, permute=False):
    cka_cor = linear_CKA(dat1, dat2)
    if permute:
        correlation_list = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = []
            for i in range(n_sample):
                dat1_perm = dat1[np.random.permutation(dat1.shape[0]), :]
                dat2_perm = dat1[np.random.permutation(dat2.shape[0]), :]
                results.append(pool.apply_async(linear_CKA, (dat1_perm, dat2_perm)))
            for result in results:
                pert_cka_cor = result.get()
                correlation_list.append(pert_cka_cor)
        p_value = np.sum(np.array(correlation_list) > cka_cor) / n_sample
        adj_cka_cor = cka_cor - np.mean(correlation_list)
    else:
        adj_cka_cor = None
        p_value = None
    return cka_cor, adj_cka_cor, p_value
    
def process_dat(dat_i, dat_j, dat_names):
    dat1_name = dat_names[dat_i]
    dat2_name = dat_names[dat_j]
    dat1, dat1_gene = read_dat(dat1_name)
    dat2, dat2_gene = read_dat(dat2_name)

    gene_inter = set(dat1_gene).intersection(set(dat2_gene))

    # print(f' {dat1_name} has {len(dat1_gene)} genes')
    # print(f' {dat2_name} has {len(dat2_gene)} genes')
    # print('intersection', len(gene_inter))

    # select the common genes
    dat1 = dat1[np.array([i for i,gene in enumerate(dat1_gene) if gene in gene_inter]), :]
    dat2 = dat2[np.array([i for i,gene in enumerate(dat2_gene) if gene in gene_inter]), :]

    # sort the data based on genes
    dat1_gene = [gene for gene in dat1_gene if gene in gene_inter]
    dat2_gene = [gene for gene in dat2_gene if gene in gene_inter]

    dat1 = dat1[np.argsort(dat1_gene), :]
    dat2 = dat2[np.argsort(dat2_gene), :]
    # dat1_gene = dat1_gene[np.argsort(dat1_gene)]
    # dat2_gene = dat2_gene[np.argsort(dat2_gene)]


    # standardize the data
    dat1 = (dat1 - np.mean(dat1, axis=0)) / np.std(dat1, axis=0)
    dat2 = (dat2 - np.mean(dat2, axis=0)) / np.std(dat2, axis=0)
    
    return dat1, dat2, dat1_name, dat2_name


def read_dat(data_type, n_dim=512):
    if data_type == 'gene2vec':
        if os.path.exists(f'{out_dir}/gene2vec.npy'):
            gene2vec = np.load(f'{out_dir}/gene2vec.npy')
            gene2vec_gene = np.load(f'{out_dir}/gene2vec_gene.npy', allow_pickle=True)
            return gene2vec, gene2vec_gene
        
        gene2vec = '/local2/yuyan/gene_emb/data/gene2vec/gene2vec_dim_200_iter_9_w2v.txt.1'
        gene2vec = pd.read_csv(gene2vec, sep=' ', header=None, skiprows=1, index_col=0)
        gene2vec_gene = gene2vec.index.values
        gene2vec = np.array(gene2vec) #(n_gene, dim)
        gene2vec = gene2vec[:, :-1] 
        np.save(f'{out_dir}/gene2vec.npy', gene2vec)
        np.save(f'{out_dir}/gene2vec_gene.npy', gene2vec_gene)

        return gene2vec, gene2vec_gene
    if data_type == 'genept':
        if os.path.exists(f'{out_dir}/genept.npy'):
            genept = np.load(f'{out_dir}/genept.npy')
            genept_gene = np.load(f'{out_dir}/genept_gene.npy')
            return genept, genept_gene
        
        genept = '/local2/yuyan/gene_emb/data/genept/data_embedding/GPT_3_5_gene_embeddings.pickle'
        genept = pickle.load(open(genept, 'rb'))
        genept_gene = list(genept.keys())
        genept = np.stack([np.array(genept[gene]) for gene in genept_gene], axis=0)

        np.save(f'{out_dir}/genept.npy', genept)
        np.save(f'{out_dir}/genept_gene.npy', genept_gene)

        return genept, genept_gene
    if data_type == 'biolinkbert_summary':
        if os.path.exists(f'{out_dir}/biolinkbert_summary.npy'):
            biolinkbert_summary = np.load(f'{out_dir}/biolinkbert_summary.npy')
            biolinkbert_summary_gene = np.load(f'{out_dir}/biolinkbert_summary_gene.npy')
            return biolinkbert_summary, biolinkbert_summary_gene
        
        biolinkbert = '/local2/yuyan/gene_emb/data/biolinkbert/BioLinkBERT_gene_summary_embeddings.pkl'
        with open(biolinkbert,'rb') as f:
            biolinkbert = pickle.load(f)
        biolinkbert_gene = list(biolinkbert['symbol'])
        biolinkbert = np.stack([np.array(emb) for emb in biolinkbert['embedding']], axis=0)
        np.save(f'{out_dir}/biolinkbert_summary.npy', biolinkbert)
        np.save(f'{out_dir}/biolinkbert_summary_gene.npy', biolinkbert_gene)

        return biolinkbert, biolinkbert_gene

    if data_type == 'biolinkbert_genename':
        if os.path.exists(f'{out_dir}/biolinkbert_genename.npy'):
            biolinkbert_genename = np.load(f'{out_dir}/biolinkbert_genename.npy')
            biolinkbert_genename_gene = np.load(f'{out_dir}/biolinkbert_genename_gene.npy')
            return biolinkbert_genename, biolinkbert_genename_gene
        
        biolinkbert = '/local2/yuyan/gene_emb/data/biolinkbert/BioLinkBERT_gene_name_embeddings.pkl'
        with open(biolinkbert,'rb') as f:
            biolinkbert = pickle.load(f)
        biolinkbert_gene = list(biolinkbert['symbol'])
        biolinkbert = np.stack([np.array(emb) for emb in biolinkbert['embedding']], axis=0)
        np.save(f'{out_dir}/biolinkbert_genename.npy', biolinkbert)
        np.save(f'{out_dir}/biolinkbert_genename_gene.npy', biolinkbert_gene)

        return biolinkbert, biolinkbert_gene

    if data_type == 'geneformer':
        if os.path.exists(f'{out_dir}/geneformer.npy'):
            geneformer = np.load(f'{out_dir}/geneformer.npy')
            geneformer_gene = np.load(f'{out_dir}/geneformer_gene.npy')
            return geneformer, geneformer_gene
        
        geneformer = '/local2/yuyan/gene_emb/code/Geneformer/gene_embedding.pkl'
        with open(geneformer,'rb') as f:
            geneformer = pickle.load(f)
        geneformer_gene = list(geneformer.keys())
        geneformer = np.stack([np.array(geneformer[k]) for k in geneformer], axis=0)
        np.save(f'{out_dir}/geneformer.npy', geneformer)
        np.save(f'{out_dir}/geneformer_gene.npy', geneformer_gene)

        return geneformer, geneformer_gene

    if data_type == 'prottrans':
        if os.path.exists(f'{out_dir}/prottrans.npy'):
            dat = np.load(f'{out_dir}/prottrans.npy')
            dat_gene = np.load(f'{out_dir}/prottrans_gene.npy')
            return dat, dat_gene
        
        dat = '/local2/yuyan/gene_emb/data/prottran/gene_2_embedding_dict.pkl'
        with open(dat,'rb') as f:
            dat = pickle.load(f)
        dat_gene = list(dat.keys())
        dat = np.stack([np.array(dat[k]) for k in dat], axis=0)
        np.save(f'{out_dir}/{data_type}.npy', dat)
        np.save(f'{out_dir}/{data_type}_gene.npy', dat_gene)

        return dat, dat_gene
    
    if data_type == 'biogpt':
        if os.path.exists(f'{out_dir}/{data_type}.npy'):
            dat = np.load(f'{out_dir}/{data_type}.npy')
            dat_gene = np.load(f'{out_dir}/{data_type}_gene.npy')
            return dat, dat_gene
        
        dat = '/local2/yuyan/gene_emb/data/biogpt/bioGPT_gene_summary_embeddings.pkl'
        with open(dat,'rb') as f:
            dat = pickle.load(f)
        dat_gene = dat['symbol']
        dat = np.array(dat['embedding'])
        
        np.save(f'{out_dir}/{data_type}.npy', dat)
        np.save(f'{out_dir}/{data_type}_gene.npy', dat_gene)

        return dat, dat_gene

    if data_type == 'omics':
        if os.path.exists(f'{out_dir}/omics.npy'):
            dat = np.load(f'{out_dir}/omics.npy')
            dat_gene = np.load(f'{out_dir}/omics_gene.npy')
            return dat, dat_gene
        
        dat = '/local2/yuyan/gene_emb/data/omics/Supplementary_Table_S3_OMICS_EMB.tsv'
        dat = pd.read_csv(dat, sep='\t')
        dat_gene = dat['gene_id'].values
        ensembl2symbol = {}
        # batchly convert ensembl ids to gene symbols with batch size = 50
        for i in tqdm(range(0, len(dat_gene), 50)):
            ensembl2symbol.update(batch_convert_ensembl_to_symbols(list(dat_gene[i:i+50])))

        dat_gene = [ensembl2symbol[gene] for gene in dat_gene]
        dat = np.array(dat.iloc[:, 1:])

        np.save(f'{out_dir}/{data_type}.npy', dat)
        np.save(f'{out_dir}/{data_type}_gene.npy', dat_gene)
    
    

    if data_type == 'string_stru2vec':
        if os.path.exists(f'{out_dir}/{data_type}.npy'):
            dat = np.load(f'{out_dir}/{data_type}.npy')
            dat_gene = np.load(f'{out_dir}/{data_type}_gene.npy')
            return dat, dat_gene
        
        dat = '/local2/yuyan/gene_emb/data/graph_emb/STRING_PPI_struc2vec_number_walks64_walk_length16_dim500.txt'
        id_map = '/local2/yuyan/gene_emb/data/graph_emb/node_list.txt'
        dat = pd.read_csv(dat, sep=' ', header=None, skiprows=1)
        dat_gene = dat.iloc[:, 0].values

        # read in id mapping
        id_map = pd.read_csv(id_map, sep='\t')
        dat_gene = [id_map[id_map['index'] == gene]['STRING_id'].values[0] for gene in dat_gene]
        dat_gene = [gene.split('.')[1] for gene in dat_gene] # ENSP id
        dat_gene_map = {i:gene for i,gene in enumerate(dat_gene)}


        ensembl2symbol = {}
        # batchly convert ensembl ids to gene symbols with batch size = 50
        for i in tqdm(range(0, len(dat_gene), 50)):
            ensembl2symbol.update(convert_ensp_to_gene_symbol(list(dat_gene[i:i+50])))

        dat_gene = [ensembl2symbol[gene] for gene in dat_gene]
        
        dat = np.array(dat.iloc[:, 1:])

        np.save(f'{out_dir}/{data_type}.npy', dat)
        np.save(f'{out_dir}/{data_type}_gene.npy', dat_gene)

        return dat, dat_gene

    if data_type == 'random':
        gene_name = np.load(f'{out_dir}/genept_gene.npy')
        n = len(gene_name)
        random_emb = np.random.randn(n, n_dim)
        return random_emb, gene_name
    
    if data_type == 'autoencoder':        
        dat = '/local2/yuyan/gene_emb/data/autoencoder/all_2023-12-06_11-18-16.npy'
        dat = np.load(dat)
        dat_gene = np.load('/local2/yuyan/gene_emb/data/autoencoder/all_2023-12-06_11-18-16_gene.npy')
        return dat, dat_gene
        

def plot_cor_matrix(cor_matrix, dat_names, fig_name):
    # add transpose
    cor_matrix = cor_matrix + cor_matrix.T
    # set diagonal to 1
    np.fill_diagonal(cor_matrix, 1)
    
    plt.imshow(cor_matrix, cmap='viridis', interpolation='nearest')
    # put text in each cell
    for i in range(len(dat_names)):
        for j in range(len(dat_names)):
            plt.text(j, i, '{:.2f}'.format(cor_matrix[i, j]), ha='center', va='center', color='w')
    plt.xticks(np.arange(len(dat_names)), dat_names, rotation=90)
    plt.yticks(np.arange(len(dat_names)), dat_names)
    plt.tight_layout()

    plt.colorbar()
    plt.savefig(f'{fig_dir}/{fig_name}_cor_matrix.png', bbox_inches='tight')
    plt.show()
    plt.close()

def fit_svcca_model(dat1, dat2, permute=False):
    dat1 = dat1.T
    dat2 = dat2.T

    svcca_cor = cal_svcca(dat1, dat2)

    if permute:
        correlation_list = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = []
            for i in range(n_sample):
                dat1_perm = dat1[:, np.random.permutation(dat1.shape[1])]
                dat2_perm = dat2[:, np.random.permutation(dat2.shape[1])]
                results.append(pool.apply_async(cal_svcca, (dat1_perm, dat2_perm)))
            for result in results:
                pert_svcca_cor = result.get()
                correlation_list.append(pert_svcca_cor)

        p_value = np.sum(np.array(correlation_list) > svcca_cor) / n_sample
        adj_svcca_cor = svcca_cor - np.mean(correlation_list)
    else:
        adj_svcca_cor = None
        p_value = None

    return svcca_cor, adj_svcca_cor, p_value

def cal_svcca(dat1, dat2):
    dat1 = dat1.astype(np.float64)
    dat2 = dat2.astype(np.float64)
    dat1 = dat1 - np.mean(dat1, axis=1, keepdims=True)
    dat2 = dat2 - np.mean(dat2, axis=1, keepdims=True)

    U1, s1, V1 = np.linalg.svd(dat1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(dat2, full_matrices=False)

    # keep top m singular values to have sum of singular values > 0.99
    m1 = np.sum(np.cumsum(np.absolute(s1)) / np.sum(np.absolute(s1)) < 0.99)
    m2 = np.sum(np.cumsum(np.absolute(s2)) / np.sum(np.absolute(s2)) < 0.99)

    svacts1 = np.dot(s1[:m1]*np.eye(m1), V1[:m1])
    svacts2 = np.dot(s2[:m2]*np.eye(m2), V2[:m2])

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    svcca_cor = np.mean(svcca_results['cca_coef1'])

    return svcca_cor


def fit_cca_model(dat1, dat2, dat1_name, dat2_name, plot_=False, permute=False):
    n_comp = min(dat1.shape[1], dat2.shape[1])
    cca = CCA(n_components=n_comp)
    cca.fit(dat1, dat2)

    X_c, Y_c = cca.transform(dat1, dat2)

    actual_correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_comp)]
    # print(actual_correlation)

    # plot a scatter plot
    if plot_:
        if not os.path.exists(f'{fig_dir}/{dat1_name}_{dat2_name}_CCA.png'):
            plt.scatter(X_c[:, 0], Y_c[:, 0], s=1)
            plt.xlabel(dat1_name)
            plt.ylabel(dat2_name)
            plt.title('correlation: {}'.format(actual_correlation))
            plt.savefig(f'{fig_dir}/{dat1_name}_{dat2_name}_CCA.png')
            plt.show()
            plt.close()
    
    actual_correlation = np.mean(np.square(actual_correlations))
    if permute:
        correlation_list = []
        for i in tqdm(range(n_sample)):
            cca = CCA(n_components=1)
            dat1_perm = dat1[np.random.permutation(dat1.shape[0]), :]
            dat2_perm = dat1[np.random.permutation(dat2.shape[0]), :]
            cca.fit(dat1_perm, dat2_perm)
            X_c, Y_c = cca.transform(dat1_perm, dat2_perm2)
            correlation = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
            correlation_list.append(correlation)
        p_value = np.sum(np.array(correlation_list) > actual_correlation) / n_sample
        # print('p_value: ', p_value)
        # # plot the histogram
        # plt.hist(correlation_list, bins=20)
        # plt.axvline(actual_correlation, color='red')
        # plt.xlabel('random correlation')
        # plt.ylabel('frequency')
        # plt.title('correlation: {}'.format(correlation))
        # plt.savefig(f'{fig_dir}/{dat1_name}_{dat2_name}_CCA_perm_hist.png')
        # plt.close()
        # print('adjusted correlation:', actual_correlation - np.mean(correlation_list))
        adjusted_correlation = actual_correlation - np.mean(correlation_list)

    if stats_test_cca:
        correlation_list = []
        for i in tqdm(range(n_sample//2)):
            cca = CCA(n_components=1)
            dat_random = np.random.randn(dat2.shape[0], dat2.shape[1])
            cca.fit(dat1, dat_random)
            X_c, Y_c = cca.transform(dat1, dat_random)
            correlation = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
            correlation_list.append(correlation)
        
        for i in tqdm(range(n_sample//2)):
            cca = CCA(n_components=1)
            dat_random = np.random.randn(dat1.shape[0], dat1.shape[1])
            cca.fit(dat_random, dat2)
            X_c, Y_c = cca.transform(dat_random, dat2)
            correlation = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
            correlation_list.append(correlation)

        p_value = np.sum(np.array(correlation_list) > actual_correlation) / n_sample
        print('p_value: ', p_value)
        # plot the histogram
        plt.hist(correlation_list, bins=20)
        plt.axvline(actual_correlation, color='red')
        plt.xlabel('random correlation')
        plt.ylabel('frequency')
        plt.title('correlation: {}'.format(correlation))
        plt.savefig(f'{fig_dir}/{dat1_name}_{dat2_name}_CCA_hist.png')
        plt.close()
        print('adjusted correlation:', actual_correlation - np.mean(correlation_list))

    return actual_correlation, adjusted_correlation if perm_test_cca else actual_correlation, None

def remove_dup(dat, dat_gene):
    # remove duplication of genes in dat_gene, only keep the first one, and remove the corresponding row in dat
    dat_gene_unique = []
    dat_unique = []
    for i, gene in enumerate(dat_gene):
        if gene not in dat_gene_unique:
            dat_gene_unique.append(gene)
            dat_unique.append(dat[i, :])
    dat_unique = np.stack(dat_unique, axis=0)
    return dat_unique, dat_gene_unique

if __name__ == '__main__':
    main()
