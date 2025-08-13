import sys
import os

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
import os
# Import feature maps
from feature_maps import IdentityLayer, RecenteringLayer, GramMatrixLayer, \
                        RecenterBondLayer, OrthogonalChangeOfBasisBatched
import torch
import diffusion_map as diffusion_map
from tqdm import tqdm

def compute_eigencoords(data, sigma):
    adj = calc_W(data, sigma, threshold=False)
    row_sums = np.sum(adj, axis=1)
    row_sums_tensor_product = np.outer(row_sums, row_sums)
    adj = adj / row_sums_tensor_product
    lap = laplacian(adj)
    phi, Sigma = calc_vars(data, adj, sigma, n_eigenvectors = 25)
    phi = phi[:,1:]
    Sigma = Sigma[1:]
    Sigma = np.sort(Sigma)
    return phi, Sigma, lap

def plot_rel_diffs(ax, Sigma, name):
    relative_differences = Sigma[1:] / Sigma[:-1]
    # relative_differences = Sigma/np.linalg.norm(Sigma)
    ax.plot([int(i+1) for i in range(relative_differences.shape[0])], \
                relative_differences, alpha=1, linestyle='-', marker = ".", label=name)

def rel_diff_plotter(feature_maps, data, max_eps_values):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name, feature_map in feature_maps.items():
        feature_map = feature_maps[name]
        feature_data = feature_map(torch.tensor(data.reshape(batch_size, 3 * N), dtype=torch.float32))
        max_eps = max_eps_values[name]
        phi, Sigma = compute_eigencoords(feature_data.detach().numpy(), sigma=max_eps)
        plot_rel_diffs(ax, Sigma, name)
    ax.set_xlabel('Index')
    ax.set_xticks([int(i+1) for i in range(Sigma.shape[0])])
    ax.set_ylabel(r'$\lambda_{i+1}/\lambda_i$') # Relative Difference
    ax.set_title('Successive Relative Differences between Singular Values')
    ax.legend()
    plt.savefig('relative_differences_singular_values.png')

def featurized_laplacian_embedding(feature_map, data, max_eps=10.0):
    feature_data = feature_map(torch.tensor(data.reshape(batch_size, 3 * N), dtype=torch.float32)).detach().numpy()
    # compute eigenvalues and eigenvectors
    dmap = diffusion_map.DiffusionMap(alpha=1.0, epsilon=max_eps, num_evecs=25)
    dmap._compute_knn_sq_dists(feature_data.T)
    dmap.construct_generator(feature_data.T)
    lap = dmap.L
    phi, _, Sigma = dmap._construct_diffusion_coords(lap)
    # phi, Sigma, lap = compute_eigencoords(feature_data.detach().numpy(), sigma=max_eps)
    return phi, Sigma, lap


def hypersurface_embedding(phi, Sigma, lap, intrinsic_dim, zeta=0.3):
    # setup
    embedding_dim = intrinsic_dim + 1
    t_bundle = RiemannMetric(phi, lap, n_dim=embedding_dim)
    H = t_bundle.get_dual_rmetric()
    principal_space = t_bundle.Hvv[...,:embedding_dim]

    # eigencoordinate selection
    candidate_dim = principal_space.shape[1]
    if intrinsic_dim == 1:
        opt_proj_axis = [0]
        before_vol = np.mean(np.log(np.abs(principal_space[:,0,0])))
        # before_vol = 0.0
    else:
        # proj_vol, all_comb = projected_volume(principal_space[...,:intrinsic_dim], \
        #                                     intrinsic_dim, eigen_values=Sigma, zeta=zeta)
        proj_vol = _comp_projected_volume(principal_space, [i for i in range(intrinsic_dim)], \
                                          intrinsic_dim, embedding_dim, eigen_values=Sigma, zeta=zeta)
        # argmax_proj_vol = proj_vol.mean(1).argmax()
        # before_vol = proj_vol.mean(1).max()
        before_vol = np.mean(proj_vol)
        opt_proj_axis = [i for i in range(intrinsic_dim)]
        # opt_proj_axis = list(all_comb[argmax_proj_vol])
        
    remaining_axes = [ii for ii in range(candidate_dim)
                    if ii not in opt_proj_axis]
    
    # add last coordinate
    # embedding_dim_vols = []
    # for i in remaining_axes:
    #     embedding_axes = sorted(opt_proj_axis + [i])
    #     vol = _comp_projected_volume(principal_space, \
    #             np.array(embedding_axes),\
    #             embedding_dim, embedding_dim, Sigma, -zeta)
    #     embedding_dim_vols.append((i,np.mean(vol)))
    # final_dim, after_vol = sorted(embedding_dim_vols, key=lambda x: x[1])[0]
    # embedding_axes = sorted(opt_proj_axis + [final_dim])

    # final processing
    embedding_axes = opt_proj_axis + [min(remaining_axes)]
    emb_vol = _comp_projected_volume(principal_space, \
            np.array(embedding_axes),\
            embedding_dim, embedding_dim, eigen_values=None, zeta=-zeta)
    after_vol = np.mean(emb_vol)
    
    return before_vol, after_vol, embedding_axes


# get data
data_dir = np.load("ground_data/butane_nonaligned.npz")
data = data_dir['data_all_atom'][::100]
dihderals = data_dir['dihedrals'][::100]
max_eps_values = np.load('max_eps_values_ksum.npz', allow_pickle=True)
# max_eps_values['GramMatrixCarbon'] = 1.0

# Example input tensor of shape [batch_size, 3N]
batch_size = data.shape[0]
N = data.shape[1] // 3 # Number of atoms

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

# intrinsic_dim = 3
# embedding_dim = intrinsic_dim + 1
N_search_dims = 5
hypersurface_data = np.zeros((len(feature_maps), N_search_dims))

for i, (name, feature_map) in enumerate(feature_maps.items()):
    eps = max_eps_values[name]
    phi, Sigma, lap = featurized_laplacian_embedding(feature_map, data, max_eps=eps)
    print(f"Feature Map: {name}")
    for intrinsic_dim in tqdm(range(1, N_search_dims+1)):
        embedding_dim = intrinsic_dim + 1
        before_vol, after_vol, embedding_axes = hypersurface_embedding(phi, -Sigma, lap, intrinsic_dim, zeta=0.01)
        if np.isnan(after_vol) or np.isnan(before_vol):
            print(f"Skipping intrinsic_dim {intrinsic_dim} for feature map {name} due to NaN values.")
            continue
        hypersurface_data[i, intrinsic_dim-1] = (before_vol - after_vol)/np.abs(before_vol) if before_vol > 1 \
                                                else (before_vol - after_vol)
        print(f"Embedding axes: {embedding_axes}")

    
# for intrinsic_dim in tqdm(range(1, N_search_dims+1)):
#     embedding_dim = intrinsic_dim + 1
#     for i, (name, feature_map) in enumerate(feature_maps.items()):
#         feature_data = feature_map(torch.tensor(data.reshape(batch_size, 3 * N), dtype=torch.float32))
#         max_eps = max_eps_values[name]
#         before_vol, after_vol, embedding_axes = hypersurface_embedding(feature_map, data, intrinsic_dim, max_eps)
#         hypersurface_data[i, intrinsic_dim-1] = (before_vol - after_vol)/np.abs(before_vol) if before_vol > 0 else before_vol - after_vol

# hypersurface_data = np.array([[-1.57841356, -0.27104339,  1.66147997,  0.024516  ,  0.15922294],
#                             [-1.66897645, -0.27849418,  1.93620565, -0.08683134,  0.29023821],
#                             [-1.73323673,  0.61634487,  0.41963689,  0.57570177,  0.65850341],
#                             [-2.09947075,  0.13443742,  1.8300723 , -1.88148133,         np.nan],
#                             [-1.77165535,  0.36810414,  2.09445835,  0.22344317, -0.07836842],
#                             [-1.5100376 ,  0.08786685,  0.23069602,  1.17615726,  0.08736866],
#                             [-1.06645842,  1.2168909 ,  1.42912508,  0.55524473, -0.47969583]])
# Plot each row in hypersurface_data
print("Hypersurface Data")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for i, (name, feature_map) in enumerate(feature_maps.items()):
    if name == 'OrthogonalChangeOfBasisBatched':
        name = 'PlaneAlign'
    ax.plot([i+2 for i in range(hypersurface_data.shape[1])], \
            hypersurface_data[i, :], label=name, \
            marker='s', linestyle='-', linewidth=3.0, markersize=6.0, alpha=1.0)

ax.set_xlabel('Embedding dimension (D+1)', fontsize=20)
ax.set_ylabel('HyperSurface(D+1)', fontsize=20)
ax.set_title('Hypersurface Data for Different Feature Maps', fontsize=20)
ax.set_xticks([i+2 for i in range(hypersurface_data.shape[1])])
ax.tick_params(axis='both', which='major', labelsize=15)
plt.legend()
plt.savefig('hypersurface_data_ksum_eps.png')
plt.show()

np.savez('hypersurface_data_full_data.npz', hypersurface_data=hypersurface_data, max_eps_values=max_eps_values)
name = 'OrthogonalChangeOfBasisBatched'
feature_map = feature_maps[name]
feature_data = feature_map(torch.tensor(data.reshape(batch_size, 3 * N), dtype=torch.float32))
max_eps = max_eps_values[name]
phi, Sigma, lap = compute_eigencoords(feature_data.detach().numpy(), sigma=10.0)
fig = plt.figure()
ax = fig.add_subplot()
s = ax.scatter(phi[:,0], -phi[:,1], c=dihderals, cmap='hsv')
c = fig.colorbar(s, location='left')
c.set_label(r'$\theta$')
ax.set_xlabel(r'$\psi_1$')
# ax.set_ylabel(r'$\psi_2$')
plt.savefig(f'2d_plot_{name}.png')
