import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import diffusion_map as diffusion_map
from sklearn.neighbors import NearestNeighbors
import torch
from feature_maps import IdentityLayer, RecenteringLayer, GramMatrixLayer, \
                        RecenterBondLayer, OrthogonalChangeOfBasisBatched
# get data
data_dir = np.load("ground_data/butane_nonaligned.npz")
data = data_dir['data_all_atom'][::100]
dihderals = data_dir['dihedrals'][::100]

# Example input tensor of shape [batch_size, 3N]
batch_size = data.shape[0]
N = data.shape[1] // 3 # Number of atoms

# Test GramMatrixLayer
selected_rows = [i for i in range(N)]  # Example: select the 1st and 3rd rows
all_atoms = torch.zeros(len(selected_rows), N)
for i, row in enumerate(selected_rows):
    all_atoms[i, row] = 1

selected_atoms = [3, 6, 9, 13]  # Example: select the first 4 atoms
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

# Loop over all feature maps and compute the ksum test for each
max_eps_values = {}
plotting_data = {}
max_stats = {}
i=0
ksum_test = False
if ksum_test:
    for name, feature_map in feature_maps.items():
        feature_data = feature_map(torch.tensor(data.reshape(batch_size, 3 * N), dtype=torch.float32)).detach().numpy()
        dmap = diffusion_map.DiffusionMap(alpha=1.0, epsilon="MAX_MIN")
        dmap._compute_knn_sq_dists(feature_data.T)
        print(f"computed sq_dists for {name}!")
        max_eps, min_eps, _ = dmap.max_min_epsilon(k_alpha=0.1)
        max_eps_values[name] = max_eps
        # max_eps, max_derivative_index, max_derivative_value, \
        #     eps_range, discrete_derivative = dmap.k_sum_test(feature_data.T)
        # # # opt_eps, eps_range, semi_group_vals = dmap.semigroup_test(feature_data.T)
        # if name == 'GramMatrixCarbon':
        #         # max_eps, _, _ = dmap.max_min_epsilon()
        #         max_eps = 1.0
        #         # print("whoop")
        # if name == 'OrthogonalChangeOfBasisBatched':
        #         max_eps, _, _ = dmap.max_min_epsilon()
        #         # max_eps = 100.0
        #         # print("whoop")
        # max_eps_values[name] = max_eps
        # plotting_data[name] = (eps_range, discrete_derivative)
        # max_stats[name] = (max_derivative_index, max_derivative_value)
        i = i+1
    print("Max epsilon values for each feature map:")
    print(max_eps_values)
    # Save the dictionary max_eps_values as an npz file
    np.savez('max_eps_values_max_min_eps.npz', **max_eps_values)

def compute_intrinsic_dim(name, max_stats, plotting_data):
    eps_range = plotting_data[name][0]
    discrete_derivative = max_stats[name][1]
    dlog_eps = np.log(eps_range[1]) - np.log(eps_range[0])
    intrinsic_dim = discrete_derivative / dlog_eps
    return intrinsic_dim 

plot_embedding = True
if plot_embedding:
    name = 'BondAlign(2,3)'  # Example feature map to plot
    feature_map = feature_maps[name]
    feature_data = feature_map(torch.tensor(data.reshape(batch_size, 3 * N), dtype=torch.float32)).detach().numpy()
    dmap = diffusion_map.DiffusionMap(alpha=1.0, num_evecs = 25, epsilon="MAX_MIN")
    dmap._compute_knn_sq_dists(feature_data.T)
    max_eps, max_derivative_index, max_derivative_value, \
           eps_range, discrete_derivative = dmap.k_sum_test(feature_data.T)
    # max_eps, _, _ = dmap.max_min_epsilon()
    dmap.epsilon = max_eps
    dmap.construct_generator(feature_data.T)
    L = dmap.get_generator()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    dmap, evecs, evals = dmap._construct_diffusion_coords(L)
    ax.scatter(dmap[:,0], dmap[:,1], dmap[:,18], c=dihderals, cmap='hsv')
    # ax.colorbar(label='Dihedral Angles')
    ax.set_xlabel(r'\psi_1')
    ax.set_ylabel(r'\psi_2')
# Plot the data in the dictionary plotting_data
plotting = False
if plotting:
    for name, (eps_range, discrete_derivative) in plotting_data.items():
        intrinsic_dim = compute_intrinsic_dim(name, max_stats, plotting_data)
        plt.figure()
        plt.plot(eps_range[1:], discrete_derivative, 'o-')
        plt.xscale('log')
        plt.xlabel(r'$\log \epsilon$')
        plt.ylabel(r'$d \log(K_{sum})$')
        plt.title(f'Discrete Derivative vs Epsilon for {name}')
        plt.legend()
        plt.savefig(f'pictures/{name}_discrete_derivative_vs_epsilon_1K_pts.png')
        plt.close()
        print('Computed for {name}')

