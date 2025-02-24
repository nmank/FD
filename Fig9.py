# only works with sum of grassmannian chordal distances rather than chordal distance between flags.
# and SVD outpreforms other methods


import numpy as np
import scipy
import scipy.io as sio

from utils import chordal_distance, truncate_svd
from FD import FD

from matplotlib import pyplot as plt

from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import tqdm

import pandas as pd
import seaborn as sns

def make_Bs(fl_type):
    Bs = [np.arange(fl_type[0])]
    for i in range(1,len(fl_type)):
        Bs.append(np.arange(fl_type[i-1],fl_type[i]))
    return Bs

def evaluate_knn_with_distances(distance_matrix_train, distance_matrix_test, y_train, y_test, k_values):
    accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
        
        # Fit the model using the training distance matrix and labels
        knn.fit(distance_matrix_train, y_train)
        
        # Predict using the test distance matrix
        y_pred = knn.predict(distance_matrix_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # print(f"k={k}, Accuracy: {accuracy:.4f}")
        accuracies.append(accuracy)
    
    return accuracies

def extract_patches_of_class(data, labels, patch_size, target_class):
    """
    Extract non-overlapping patches where all pixels in the patch are of the target class.

    :param data: The hyperspectral image data 
    :param labels: The ground truth labels 
    :param patch_size: Size of the patch (e.g., 7 for 7x7 patches).
    :param target_class: The class for which patches should be extracted.
    :return: A list of patches (each patch is of size patch_size x patch_size x num_bands).
    """
    half_patch = patch_size // 2
    patches = []
    patch_labels = []

    # Iterate through the image in steps of patch_size to avoid overlap
    for i in range(half_patch, data.shape[0] - half_patch, patch_size):
        for j in range(half_patch, data.shape[1] - half_patch, patch_size):
            # Extract the patch from both the data and the labels
            label_patch = labels[i - half_patch:i + half_patch + 1, j - half_patch:j + half_patch + 1]
            
            # Check if all pixels in the label patch are of the target class
            if np.all(label_patch == target_class):
                # Extract the corresponding data patch
                patch = data[i - half_patch:i + half_patch + 1, j - half_patch:j + half_patch + 1, :]
                patches.append(patch)
                patch_labels.append(target_class)

    return np.array(patches), np.array(patch_labels)

def extract_patches(data, labels, patch_size, class_ids):
    # extract patches
    mod_data = []
    mod_labels = []
    for target_class in class_ids:
        patches, patch_labels = extract_patches_of_class(data, labels, patch_size, target_class)
        if len(patches) > 0:
            flat_patches = []
            for patch in patches:
                # Your 3D array of size 11x11x200
                array_3d = patch  # Example array

                center_x, center_y = patch_size//2, patch_size//2

                # Create a list of all (x, y) coordinates and compute their Manhattan distances from the center
                coords = [(x, y) for x in range(patch_size) for y in range(patch_size)]
                distances = [(x, y, max(abs(x - center_x), abs(y - center_y))) for x, y in coords]

                # Sort coordinates by distance
                sorted_coords = sorted(distances, key=lambda item: item[2])

                # Create the 2D array by unwrapping the 3D array based on sorted coordinates
                flat_patch = np.array([array_3d[x, y, :] for x, y, _ in sorted_coords])
                flat_patches.append(flat_patch.T)

                # Create a hierarchy vector containing the Chebyshev distances in the same sorted order
                hierarchy_vector = np.array([distance for _, _, distance in sorted_coords])

                # Find the indices where the hierarchy vector changes value
                change_indices = np.where(np.diff(hierarchy_vector) != 0)[0] + 1  # Add 1 because diff reduces length by 1

            change_indices = np.hstack([change_indices,np.array(len(hierarchy_vector))])
            mod_labels +=[target_class]*len(patches)
            
            mod_data += flat_patches

            Aset = [np.arange(i) for i in change_indices]

        else:
            print(f'No patches of class id {target_class}')

    
        print(f"Extracted {len(patches)} patches where all pixels are of class {class_names[target_class]}. Each patch has shape {patch_size}.")

    return mod_data, mod_labels, Aset


if __name__ == '__main__':

    data = scipy.io.loadmat('../data/KSC/KSC.mat')['KSC']
    labels = scipy.io.loadmat('../data/KSC/KSC_gt.mat')['KSC_gt']


    plt.figure()
    plt.imshow(data[:,:,40], cmap = 'grey')
    plt.axis('off')
    plt.savefig('../results/KSC_1band.pdf', bbox_inches = 'tight')

    class_names = {1: 'Scrub',
                2: 'Willow swamp',
                3: 'Cabbage palm hammock',
                4: 'Cabbage palm/oak hammock',
                5: 'Slash pine',
                6: 'Oak/broad leaf hammock',
                7: 'Hardwood swamp',
                8: 'Graminoid marsh',
                9: 'Spartina marsh',
                10: 'Cattail marsh',
                11: 'Salt marsh',
                12: 'Mudflats',
                13: 'Water'}
    class_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13]

    patch_size = 3
    k_values = np.arange(1,25)

    fl_type = [1,8]

    n_trials = 20

    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow-Green
        "#17becf",  # Teal
        "#aec7e8",  # Light Blue
        "#ffbb78",  # Light Orange
        "#98df8a",  # Light Green
        "#ff9896",  # Light Red
        "#c5b0d5",  # Light Purple
        "#c49c94",  # Light Brown
    ]

    methods = ['FD', 'QR', 'SVD',  'Euclidean']

    dist_mats = {}
    flag_data = {}
    flag_types = {}
    
    mod_data, mod_labels, Aset = extract_patches(data, labels, patch_size, class_ids)

    n,p = mod_data[0].shape

    n_pts = len(mod_data)

    for method_name in methods:
        print(f'Starting method {method_name}')
        # make the flags
        flag_data[method_name] = []
        flag_types[method_name] = []
        for pt in tqdm.tqdm(mod_data):
            if method_name == 'FD':
                my_flag_rep = FD(Aset=Aset, solver='svd', flag_type = fl_type)
                flag_pt, _ = my_flag_rep.fit_transform(pt)
                flag_types[method_name].append(fl_type)
            elif method_name == 'SVD':
                flag_pt = truncate_svd(pt,fl_type[-1])
                flag_types[method_name].append(fl_type)
            elif method_name == 'QR':
                Q,_ = np.linalg.qr(pt)
                flag_pt = Q[:,:fl_type[-1]]
                flag_types[method_name].append(fl_type)
            elif method_name == 'Euclidean':
                flag_pt = flag_pt.flatten()
            flag_data[method_name].append(flag_pt)

        #make distance matrices
        dist_mats[method_name] = np.zeros((n_pts,n_pts))
        Bs = make_Bs(fl_type)
        for i in tqdm.tqdm(range(n_pts)):
            for j in range(i+1,n_pts):
                x = flag_data[method_name][i]
                y = flag_data[method_name][j]
                if method_name == 'Euclidean':
                    dist = np.linalg.norm(x-y)
                else:
                    dist = chordal_distance(x, y, Bs, Bs, manifold='Grassmann')
                dist_mats[method_name][i,j] = dist
                dist_mats[method_name][j,i] = dist
            
    results = pd.DataFrame(columns = ['k','Method Name', 'Accuracy', 'Seed'])

    indices = np.arange(len(mod_labels))
    mod_labels = np.array(mod_labels)

    for s in range(n_trials):

        # Step 2: Perform train-test split based on labels using the indices
        train_indices, test_indices, _, _ = train_test_split(indices, mod_labels, test_size=0.3, stratify=mod_labels, random_state=s)

        # Step 3: Use these indices to retrieve the corresponding data and labels
        # (This step assumes `data` is an array of the same length as `labels`)
        for method_name in methods[:-1]:

            distance_matrix_train = dist_mats[method_name][train_indices,:][:,train_indices]
            distance_matrix_test = dist_mats[method_name][test_indices,:][:,train_indices]
            y_train = mod_labels[train_indices]
            y_test = mod_labels[test_indices]

            # Step 5: Test for different values of k (number of neighbors)

            accs = evaluate_knn_with_distances(distance_matrix_train, distance_matrix_test, y_train, y_test, k_values)

            for k, acc in zip(k_values, accs):
                res = pd.DataFrame(columns = results.columns,
                                data = [[k, method_name, acc, s]])
                results = pd.concat([results,res])
    
    results.to_csv('../results/ksc_robust_res.csv')

    results['Method'] = results['Method Name']
    results.sort_values(by='Method')

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize = (9,3))
    sns.lineplot(data = results, x = 'k', y = 'Accuracy', hue = 'Method', 
                 palette={'FD':'tab:blue', 'SVD':'tab:green', 'QR': 'tab:red'},
                 style = 'Method')
    # Position the legend inside the plot (upper-right corner)
    plt.legend(loc='upper right', bbox_to_anchor=(0.6, 0.6), title="")
    plt.tight_layout()
    # plt.show()
    plt.savefig('../results/KSC.pdf', bbox_inches = 'tight')


