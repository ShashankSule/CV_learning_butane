import sys
import os
import yaml  # For reading YAML configuration files

# Add the src and src/ies directories to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/ies')))

from methods import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from algorithm import factorize
from coord_search import *
from coord_search import _comp_projected_volume, projected_volume
from rmetric import RiemannMetric
from scipy.sparse.csgraph import laplacian
from utils import calc_W, calc_vars
from param_tools import r_surface
import diffusion_map
import os

# Import feature maps
from feature_maps import IdentityLayer, RecenteringLayer, GramMatrixLayer, \
                        RecenterBondLayer, OrthogonalChangeOfBasisBatched
import torch
from tqdm import tqdm

from datasets import ground_data

# Function to read configuration
def read_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def process_feature_maps(feature_map, N):
    # Test GramMatrixLayer
    selected_rows = [i for i in range(N)]  # Example: select the 1st and 3rd rows
    all_atoms = torch.zeros(len(selected_rows), N)
    for i, row in enumerate(selected_rows):
        all_atoms[i, row] = 1

    selected_atoms = [3, 6, 9, 13]  # Example: select the carbons
    carbons_matrix = torch.zeros(len(selected_atoms), N)
    for i, atom in enumerate(selected_atoms):
        carbons_matrix[i, atom] = 1

    feature_maps = {
        'NoFeaturzation': IdentityLayer(),
        'Recentering': RecenteringLayer(all_atoms),
        'GramMatrix': GramMatrixLayer(all_atoms),
        'GramMatrixCarbon': GramMatrixLayer(carbons_matrix),
        'BondAlign(1,2)': RecenterBondLayer([3, 6], batch_mode=True),
        'BondAlign(2,3)': RecenterBondLayer([6, 9], batch_mode=True),
        'OrthogonalChangeOfBasisBatched': OrthogonalChangeOfBasisBatched()
    }
    
    if feature_map not in feature_maps.keys():
        print(f"Feature map '{feature_map}' not recognized. Available options are: {list(feature_maps.keys())}")
        return None
    else:
        return feature_maps[feature_map]

def process_epsilon(feature_map):
    max_eps_values = np.load('max_eps_values.npz', allow_pickle=True)
    if feature_map in max_eps_values.keys():
        return max_eps_values[feature_map]
    else:
        return 1.0
    

if __name__ == "__main__":
        # Read configuration file
    config_file = "config_dmap.yml"  # Specify the path to your YAML config file
    config = read_config(config_file)
    feature_map_name = config["parameters"]["feature_map"]
    subsamping_rate= config["subsampling_rate"][0]
    n_evecs = config["parameters"]["n_evecs"]

    # Read data
    data_class = ground_data('ground_data/butane_nonaligned.npz', \
                        subsamping_rate, reference_CV = 'dihedrals')
    data, ref_vec = data_class.return_data()
    batch_size, N = data.shape[0], data.shape[1] // 3  # Number of atoms

    # Extract feature map name from the config file
    feature_map = process_feature_maps(feature_map_name, N)
    if feature_map is None:
        print("Invalid feature map. Exiting.")
        exit(1)
    epsilon = process_epsilon(feature_map_name)
    # epsilon = 50.0

    # Compute diffusion map 
    feature_data = feature_map(torch.tensor(data.reshape(batch_size, 3 * N), \
                                            dtype=torch.float32))
    
    selected_atoms = [3, 6, 9, 13]  # Example: select the carbons
    carbons_matrix = torch.zeros(len(selected_atoms), N)
    for i, atom in enumerate(selected_atoms):
        carbons_matrix[i, atom] = 1
    
    feature_data = feature_data.reshape((batch_size, N, 3))
    feature_data = feature_data[:, selected_atoms, :].reshape(batch_size, -1)

    dmap = diffusion_map.DiffusionMap(alpha=1.0, epsilon=epsilon, \
                                      num_evecs=n_evecs)
    dmap.construct_generator(feature_data.detach().numpy().T)
    L = dmap.L
    diff_map, evecs, evals = dmap._construct_diffusion_coords(L)

    # report statistics
    print("Data shape: ", data.shape, "feature data shape: ", feature_data.shape)
    print("Psi shape: ", diff_map.shape, "lambda shape: ", evals.shape)

    # visualize
    if n_evecs == 2:
        plt.scatter(diff_map[:, 0], diff_map[:, 1], c=ref_vec, cmap='hsv')
    else: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(diff_map[:, 0], diff_map[:, 1], diff_map[:, 2], c=ref_vec, cmap='hsv')

    # save data
    filename = 'dmaps/' + feature_map_name + \
        '_N_' + f'{batch_size}' + \
        '_epsilon_' + f'{epsilon:.1f}' + '_carbons' + '.npz'
    L_dense = L.toarray()
    np.savez(filename, feature_data=feature_data.detach().numpy(), \
            laplacian=L_dense,\
            eigvals=evals, \
            diff_map=diff_map,
            reference_CV = ref_vec,
            epsilon = epsilon,
            configs = config)


