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
    noise_exps = range(0,1000,100)
    distributions = ['Normal', 'Exponential', 'Uniform']

    linestyles = ['solid', 'dashed', 'dotted']


    Bs = make_Bs(fl_type)


    results = pd.DataFrame(columns = ['Method','Chordal Distance', 'Noise', 'Noise Dist',  'LRSE', 'SNR (dB)'])


    for noise_dist in distributions:
        for seed_num in range(n_trials):
            for noise_exp in noise_exps:
                if noise_exp > 0:
                    np.random.seed(seed_num)
                    
                    noise_fraction = compute_noise_fraction(noise_exp) 
                    
                    D, D_true, hidden_X, snr = generate_data_noise(noise_fraction, noise_dist, n, col_ids, hidden_ms)
                    
                    # FlagRep
                    my_flag_rep = FD(Aset = Aset, flag_type=fl_type, solver = 'svd')
                    X_flagrep, R_flagrep = my_flag_rep.fit_transform(D)
                    rec_flagrep = my_flag_rep.inverse_transform(X_flagrep, R_flagrep)
                    cdist_flagrep = chordal_distance(hidden_X, X_flagrep, Bs, Bs)
                    mse_flagrep = relative_log_mse(rec_flagrep,D_true)
                    row = pd.DataFrame(columns = results.columns,
                                        data = [['FD',cdist_flagrep, noise_fraction, noise_dist,  mse_flagrep, snr]])
                    results = pd.concat([results,row])

                    # Robust FlagRep
                    my_rflag_rep = FD(Aset = Aset, flag_type=fl_type, solver = 'irls svd')
                    X_rflagrep, R_rflagrep = my_rflag_rep.fit_transform(D)
                    rec_rflagrep = my_rflag_rep.inverse_transform(X_rflagrep, R_rflagrep)
                    cdist_rflagrep = chordal_distance(hidden_X, X_rflagrep, Bs, Bs)
                    mse_rflagrep = relative_log_mse(rec_rflagrep,D_true)

                    row = pd.DataFrame(columns = results.columns,
                                        data = [['RFD',cdist_rflagrep, noise_fraction, noise_dist, mse_rflagrep, snr]])
                    results = pd.concat([results,row])

                    # SVD
                    X_svd = np.linalg.svd(D)[0][:,:fl_type[-1]]
                    cdist_svd = chordal_distance(hidden_X, X_svd, Bs, Bs)
                    rec_svd = X_svd @ X_svd.T @ D
                    mse_svd = relative_log_mse(rec_svd,D_true)
                    row = pd.DataFrame(columns = results.columns,
                                        data = [['SVD',cdist_svd, noise_fraction, noise_dist, mse_svd, snr]])
                    results = pd.concat([results,row])
        
    results.to_csv('../results/noise.csv')


    plt.rcParams.update({'font.size': 14})
    # Define a list of line styles to cycle through
    line_styles = ['-', '--', ':']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    # Create the subplots
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(2, 3, figsize=(8,3), sharey='row', sharex='col')

    # Loop through different noise distributions
    for i, noise_dist in enumerate(distributions):
        idx = (results['Noise Dist'] == noise_dist) & (results['Noise'] > 0)
        
        # Non-linear line of best fit for 'Chordal Distance' by 'SNR (dB)'
        for j, method in enumerate(results['Method'].unique()):
            method_idx = idx & (results['Method'] == method)
            subset = results[method_idx]
            
            # Perform a quadratic curve fit
            popt, _ = curve_fit(quadratic, subset['SNR (dB)'], subset['Chordal Distance'])
            x_vals = np.linspace(subset['SNR (dB)'].min(), subset['SNR (dB)'].max(), 100)
            y_vals = quadratic(x_vals, *popt)
            
            # Plot the non-linear line with scatter points
            sns.scatterplot(
                data=subset, x='SNR (dB)', y='Chordal Distance', 
                ax=ax[0, i], s=2, alpha=0.2
            )
            line = ax[0, i].plot(x_vals, y_vals, linestyle=line_styles[j % len(line_styles)], alpha=0.6, lw=2)
            
        ax[0, 0].set_ylabel('Dist')
        ax[0, i].set_title(noise_dist)
        
        # Non-linear line of best fit for 'LRSE' by 'SNR (dB)'
        for j, method in enumerate(results['Method'].unique()):
            method_idx = idx & (results['Method'] == method)
            subset = results[method_idx]
            
            # Perform a quadratic curve fit
            popt, _ = curve_fit(quadratic, subset['SNR (dB)'], subset['LRSE'])
            y_vals = quadratic(x_vals, *popt)
            
            # Plot the non-linear line with scatter points
            sns.scatterplot(
                data=subset, x='SNR (dB)', y='LRSE', 
                ax=ax[1, i], s=2, alpha=0.2
            )
            line = ax[1, i].plot(x_vals, y_vals, linestyle=line_styles[j % len(line_styles)], alpha=0.6, lw=2)

    # Remove previous legends and create new handles for lines
    for a in ax.flat:
        if a.get_legend():
            a.get_legend().remove()

    # Get handles and labels from the lines of best fit
    handles, labels = [], []
    # for i, noise_dist in enumerate(distributions):
    for j, method in enumerate(results['Method'].unique()):
        # Create a Line2D object for each line to be included in the legend
        line = Line2D([0], [0], color=colors[j], linestyle=line_styles[j % len(line_styles)], lw=2)
        handles.append(line)
        labels.append(method)


    # Add a single legend inside the figure
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.53, 0.5), title="", prop={'size': 10})

    plt.tight_layout()
    plt.savefig(f'../results/noise.pdf', bbox_inches='tight')



