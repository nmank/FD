import numpy as np
from FD import FD
from matplotlib import pyplot as plt
from utils import chordal_distance, make_Bs

import pandas as pd

import seaborn as sns

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances

from sklearn.decomposition import PCA

import itertools

from sklearn.manifold import MDS


if __name__ == '__main__':

    n_pts = 60
    n_clusters = 3

    np.random.seed(42)

    n = 10
    col_ids = [20,20]
    p = np.sum(col_ids)
    hidden_flag_type = [2,4]
    hidden_ms = [2,2]
    noise_exps = range(1,1000,50)
    distributions = ['Normal', 'Exponential', 'Uniform']

    n_k = hidden_flag_type[-1]


    # generate data
    centers = []
    for _ in range(n_clusters):
        center = np.linalg.qr(np.random.randn(n, n_k))[0][:,:n_k]
        centers.append(center)

    Ds = []
    snr = []
    labels = []
    for class_num, center in enumerate(centers):
        for _ in range(n_pts//n_clusters):
            D_true = []
            for i in range(len(hidden_flag_type)): 
                cols = col_ids[i]
                n_i = hidden_flag_type[i]
                D_true_i = center[:,:n_i]@np.random.normal(size= (n_i, cols))
                D_true.append(D_true_i)
            D_true = np.hstack(D_true)
            epsilon = np.random.normal(scale = .95, size=(n,p))
            Ds.append(D_true+epsilon)
            labels.append(class_num)
            snr.append(10*np.log10(np.linalg.trace(D_true@D_true.T)/np.linalg.trace(epsilon@epsilon.T)))

    mean_snr = np.mean(snr)
    print(f'The mean SNR: {mean_snr}')


    #extract flags and euclidean data
    flags = {}
    flags['SVD'] = []
    flags['FD'] = []
    for D in Ds:
        my_flag_rep = FD(Aset = [np.arange(col_ids[0]),np.arange(p)], solver = 'svd', flag_type=[2,4])
        flagrep_X, flagrep_R = my_flag_rep.fit_transform(D)

        U = np.linalg.svd(D)[0][:,:4]            
        svd_errs = []
        col_orderings = []
        for comb in itertools.combinations([0,1,2,3], 2):
            my_flag_rep = FD(Aset = [np.arange(col_ids[0]),np.arange(p)], flag_type=[2,4])
            id1 = np.array(list(set([0,1,2,3]).difference(set(list(comb)))))
            col_ordering = np.hstack([id1,np.array(comb)])
            svd_errs.append(my_flag_rep.objective_value(U[:,col_ordering],D))
            col_orderings.append(col_ordering)
        best_idx = np.argmin(svd_errs)
        err_svd = svd_errs[best_idx]
        svd_X = U[:,col_orderings[best_idx]]
        

        flags['SVD'].append(svd_X)
        flags['FD'].append(flagrep_X)

    flags['Euclidean'] = [D.flatten() for D in Ds]




    # Distance matrices
    Bs = make_Bs(hidden_flag_type)
    dist_mats = {}
    dist_mats['SVD'] = np.zeros((n_pts,n_pts))
    dist_mats['FD'] = np.zeros((n_pts,n_pts))
    dist_mats['Euclidean'] = np.zeros((n_pts,n_pts))
    for method_name in ['Euclidean', 'SVD', 'FD']:
        for i in range(n_pts):
            for j in range(i+1,n_pts):
                x = flags[method_name][i]
                y = flags[method_name][j]
                if method_name == 'Euclidean':
                    dist = np.linalg.norm(x-y)
                else:
                    dist = chordal_distance(x, y, Bs, Bs)
                dist_mats[method_name][i,j] = dist
                dist_mats[method_name][j,i] = dist




    #Run MDS
    Ds_reduced = {}
    for method_name in ['Euclidean', 'SVD','FD']:
        my_mds = MDS(dissimilarity = 'precomputed')
        Ds_reduced[method_name] = my_mds.fit_transform(dist_mats[method_name])


    # Plotting
    fig, ax = plt.subplots(2,3, figsize = (6,4))
    for i, method_name in enumerate(['Euclidean', 'SVD', 'FD']):
        ax[0,i].imshow(dist_mats[method_name], cmap = 'grey')
        ax[0,i].set_title(method_name)
        ax[0,i].axis('off')
        for l in np.unique(labels):
            idx = np.where(labels == l)[0]
            ax[1,i].scatter(Ds_reduced[method_name][idx,0], Ds_reduced[method_name][idx,1])
            ax[1,i].set_xlabel('MDS 1')
    ax[1,0].set_ylabel('MDS 2')
    plt.tight_layout()
    plt.savefig('../results/synthetic_clustering_dr.pdf', bbox_inches = 'tight')