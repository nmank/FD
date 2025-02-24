import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from scipy.optimize import curve_fit

from FD import FD

from utils import *

# Define a quadratic function for non-linear fitting
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


if __name__ == '__main__':

    n_trials = 100
    n = 10
    col_ids = [20,20]
    p = np.sum(col_ids)
    Aset = [np.arange(col_ids[0]),np.arange(p)]
    fl_type = [2,4]
    hidden_ms = [2,2]
    distributions = ['Normal', 'Exponential', 'Uniform']
    outlier_exps = range(0,10)
    Bs = make_Bs(fl_type)
    linestyles = ['solid', 'dashed', 'dotted']






    results = pd.DataFrame(columns = ['Method','Chordal Distance', 'Proportion Outliers', 'LRSE'])
    for seed_num in range(n_trials):
        for outlier_exp in outlier_exps:
            np.random.seed(seed_num)
            prop_outliers = compute_outlier_prop(outlier_exp)
            
            D, hidden_Xs, inlier_ids, outlier_ids = generate_data_outliers(prop_outliers, n, col_ids, hidden_ms)
            D_inliers = D[:,inlier_ids]
            n_inliers = len(inlier_ids)
            


            my_flag_rep = FD(Aset = Aset, flag_type=fl_type)
            X_flagrep, R_flagrep = my_flag_rep.fit_transform(D)
            D_flagrep = X_flagrep @ X_flagrep.T @ D
            D_flagrep_inliers = D_flagrep[:,inlier_ids]
            cdist_flagrep = chordal_distance(hidden_Xs,X_flagrep,Bs,Bs)
            mse_flagrep = relative_log_mse(D_flagrep_inliers, D_inliers)


            row = pd.DataFrame(columns = results.columns,
                                data = [['FD',cdist_flagrep, prop_outliers, mse_flagrep]])
            results = pd.concat([results,row])


            my_flag_rep = FD(Aset = Aset, flag_type=fl_type, solver = 'irls svd')
            X_rflagrep, R_rflagrep = my_flag_rep.fit_transform(D)
            D_rflagrep = X_rflagrep @ X_rflagrep.T @ D
            D_rflagrep_inliers = D_rflagrep[:,inlier_ids]
            cdist_rflagrep = chordal_distance(hidden_Xs, X_rflagrep,Bs,Bs)
            mse_rflagrep = relative_log_mse(D_rflagrep_inliers, D_inliers)
            row = pd.DataFrame(columns = results.columns,
                                data = [['RFD',cdist_rflagrep, prop_outliers, mse_rflagrep]])
            results = pd.concat([results,row])

            X_svd = np.linalg.svd(D)[0][:,:fl_type[-1]]
            cdist_svd = chordal_distance(hidden_Xs, X_svd,Bs,Bs)
            D_svd_inliers = X_svd @ X_svd.T @ D_inliers
            mse_svd= relative_log_mse(D_svd_inliers, D_inliers)
            row = pd.DataFrame(columns = results.columns,
                                data = [['SVD',cdist_svd, prop_outliers, mse_svd]])
            results = pd.concat([results,row])

            X_irls_svd = irls_svd(D,fl_type[-1])
            cdist_irls_svd = chordal_distance(hidden_Xs, X_irls_svd,Bs,Bs)
            D_irls_svd_inliers = X_irls_svd @ X_svd.T @ D_inliers
            mse_irls_svd= relative_log_mse(D_irls_svd_inliers, D_inliers)
            row = pd.DataFrame(columns = results.columns,
                                data = [['IRLS-SVD',cdist_irls_svd, prop_outliers, mse_irls_svd]])
            results = pd.concat([results,row])

        
    results.to_csv('../results/outliers.csv')

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1,2,  figsize = (8,3))

    sns.lineplot(results, x = 'Proportion Outliers', y = 'Chordal Distance', hue = 'Method', ax = ax[0], style = 'Method')
    sns.lineplot(results, x = 'Proportion Outliers', y = 'LRSE', hue = 'Method', ax = ax[1], style = 'Method')

    ax[0].set_ylabel('Dist')

    for a in ax.flat:
        a.legend_.remove()

    # Get handles and labels from one of the axes
    handles, labels = ax[0].get_legend_handles_labels()

    # Add a single legend inside the figure
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.97, 0.66), title="", prop={'size': 12})


    plt.tight_layout()
    plt.savefig(f'../results/outliers.pdf', bbox_inches = 'tight')