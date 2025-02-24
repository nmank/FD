import numpy as np
from FD import FD
from matplotlib import pyplot as plt

import scipy.io as sio

from tqdm import tqdm

import pandas as pd

from utils import relative_log_mse

import seaborn as sns

if __name__ == '__main__':

    fl_type = [8,9,10]
    height, width = (50,50)
    distributions = ['Normal', 'Exponential', 'Uniform']

    results = pd.DataFrame(columns = ['Method', 'SNR (dB)', 'Added Noise', 'LRSE', 'DataSet', 'Distribution'])

    for data_name in ['KSC', 'Indian Pines']:
        if data_name == 'Indian Pines':
            # Load the hyperspectral image and ground truth
            data = sio.loadmat('../data/indian_pines/Indian_pines_corrected.mat')['indian_pines_corrected'] 
            labels = sio.loadmat('../data/indian_pines/Indian_pines_gt.mat')['indian_pines_gt']  # Shape: (145, 145)
        elif data_name == 'KSC':
            data = sio.loadmat('../data/KSC/KSC_corrected.mat')['KSC']
            labels = sio.loadmat('../data/KSC/KSC_gt.mat')['KSC_gt']

        image_height, image_width, n_bands = data.shape
        
        As = [np.arange(40), np.arange(100), np.arange(n_bands)]

        for noise_dist in distributions:

            for noise_scale in tqdm(np.arange(1,20)):
                noise_scale = .1*noise_scale
                
                for r_seed in range(20):
                    np.random.seed(r_seed)
                    # Assume `image` is a NumPy array with shape (height, width, channels)
                    
                    # Define the crop size
                    crop_height = height
                    crop_width = width

                    # Randomly select the starting point for the crop
                    x_start = np.random.randint(0, image_width - crop_width + 1)
                    y_start = np.random.randint(0, image_height - crop_height + 1)

                    # Crop the image
                    cropped_image = data[y_start:y_start + crop_height, x_start:x_start + crop_width]

                    n = width*height
                    D_true = cropped_image.reshape((width*height, n_bands))

                    D_true = D_true/D_true.max() #max normalization

                    if noise_dist == 'Normal':
                        noise = np.random.normal(scale = noise_scale, size = (n,n_bands))
                    elif noise_dist == 'Exponential':
                        noise = np.random.exponential(scale = noise_scale, size = (n,n_bands))
                    elif noise_dist == 'Uniform':
                        noise = noise_scale*np.random.uniform(size = (n,n_bands))

                    D_noisy = D_true + noise

                    snr = 10*np.log10(np.linalg.trace(D_true@D_true.T)/np.linalg.trace(noise@noise.T))

                    U, _, _ = np.linalg.svd(D_noisy)
                    D_svd = U[:,:fl_type[-1]] @ U[:,:fl_type[-1]].T @ D_noisy
                    svd_lrse = relative_log_mse(D_svd,D_true)
                    row = pd.DataFrame(columns = results.columns, data = [['SVD', snr, noise_scale, svd_lrse, data_name, noise_dist]])
                    results = pd.concat([results, row])

                    my_fr = FD(As, flag_type = fl_type)
                    X,R = my_fr.fit_transform(D_noisy)
                    D_flagrep = X @ R
                    flagrep_lrse = relative_log_mse(D_flagrep,D_true)
                    row = pd.DataFrame(columns = results.columns,  data = [['FD', snr, noise_scale, flagrep_lrse,  data_name, noise_dist]])
                    results = pd.concat([results, row])


    results.to_csv('../results/HSI_denoising.csv')

    # Load data
    results = pd.read_csv('../results/HSI_denoising.csv')
    results = results.replace({'FlagRep':'FD'})

    normal_results = results[results['Distribution'] == 'Normal'] # change to gamma or exponential to see other distributions
    normal_results = normal_results[(normal_results['SNR (dB)'] < 0) & (normal_results['SNR (dB)'] > -15)]
    normal_results = normal_results.sort_values(by = 'Method')

    fig, ax = plt.subplots(1,2,  figsize = (6,2), sharey = True)

    subset = normal_results[normal_results['DataSet'] == 'Indian Pines']
    sns.lineplot(subset, x = 'SNR (dB)', y = 'LRSE', hue = 'Method', ax = ax[0], style = 'Method', palette = {'FD': 'tab:blue', 'SVD': 'tab:green'})
    ax[0].set_title('Indian Pines')

    subset = normal_results[normal_results['DataSet'] == 'KSC']
    sns.lineplot(subset, x = 'SNR (dB)', y = 'LRSE', hue = 'Method', ax = ax[1], style = 'Method', palette = {'FD': 'tab:blue', 'SVD': 'tab:green'})
    ax[1].set_title('KSC')


    for a in ax.flat:
        a.legend_.remove()

    # Get handles and labels from one of the axes
    handles, labels = ax[0].get_legend_handles_labels()

    # Add a single legend inside the figure
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.98, 0.83), title="", prop={'size': 10})
    plt.tight_layout()

    # Display and save the figure
    plt.savefig('../results/RS_Reconstruction.pdf', bbox_inches='tight')



