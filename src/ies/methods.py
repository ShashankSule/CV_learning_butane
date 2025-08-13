# Authors: Jesse He <jeh020@ucsd.edu>
#          Tristan Brugere <tbrugre@ucsd.edu>

import numpy as np
from algorithm import factorize
from coord_search import greedy_coordinate_search
from rmetric import RiemannMetric
from scipy.sparse.csgraph import laplacian
from utils import calc_W, calc_vars
# import matlab.engine
import networkx as nx

def get_from_dict(d, values, sep=","):
    return [d[i.strip()] for i in values.split(sep)]

def sample_strip(n_samples, ratio = [1,1], seed=255):
    np.random.seed(seed)
    x = np.random.rand(n_samples) * ratio[0]
    y = np.random.rand(n_samples) * ratio[1]
    samples = np.stack([x,y]).T
    return samples

def sample_sphere(n_samples, radius=1, seed=255):
    np.random.seed(seed)
    theta = np.random.rand(n_samples) * 2 * np.pi
    phi = np.random.rand(n_samples) * np.pi - np.pi / 2
    x = np.cos(theta) * np.cos(phi) * radius
    y = np.sin(theta) * np.cos(phi) * radius
    z = np.sin(phi) * radius
    samples = np.stack([x,y,z]).T
    return samples

def sample_cylinder(n_samples, radius=1, height = 1, seed=255):
    np.random.seed(seed)
    theta = np.random.rand(n_samples) * 2 * np.pi
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    z = np.random.rand(n_samples) * height
    samples = np.stack([x,y,z]).T
    return samples

def sample_torus(n_samples, ratio = [1,1], seed=255):
    np.random.seed(seed)
    theta = np.random.rand(n_samples) * 2 * np.pi
    phi = np.random.rand(n_samples) * 2 * np.pi
    x = np.cos(theta) * ratio[0]
    y = np.sin(theta) * ratio[0]
    z = np.cos(phi) * ratio[1]
    w = np.sin(phi) * ratio[1]
    samples = np.stack([x,y,z,w]).T
    return samples

def add_noise(samples, sd = 0.5, seed=255):
    np.random.seed(seed)
    noise = np.random.normal(0, sd, (samples.shape[0],1))
    return np.hstack([samples, noise])

def ic_pml(samples, dim, sigma=0.5, n_eigenvectors=100, n_coords = 20, n_ind_coords = 10, zeta = 0, eig_crit=1.0, sim_crit=0.6,
                         self_tuning = False, k_neighbors = 7, threshold=False):
    adj = calc_W(samples, sigma=sigma, threshold=threshold)
    lap = laplacian(adj)
    phi, Sigma = calc_vars(samples, adj, sigma, n_eigenvectors = n_coords+1)
    phi = phi[:,1:]
    Sigma = Sigma[1:]
    t_bundle = RiemannMetric(phi, lap, dim)
    ind_coords = [v+1 for v in greedy_coordinate_search(
        t_bundle.get_dual_rmetric(), intrinsic_dim=dim, eigen_values=Sigma, zeta = zeta, candidate_dim = n_ind_coords)]
    results = factorize(data=samples, sigma=sigma, n_eigenvectors=n_eigenvectors, n_factors=2, eig_crit = eig_crit,
        sim_crit = sim_crit, seed=255, ind_eigs = ind_coords, self_tuning=self_tuning, k_neighbors=k_neighbors, threshold=threshold)
    data, phi, Sigma, best_matches, best_sims, C_matrix, manifolds = \
        get_from_dict(results, "data, phi, Sigma, best_matches, best_sims, C_matrix, manifolds")
    factorization = nx.DiGraph(best_matches)
    primes = [node for node in factorization.nodes if factorization.out_degree(node) == 0]
    new_manifolds = manifolds
    for i in range(len(manifolds)):
        new_manifolds[i] = [v for v in ind_coords if v in primes and v in manifolds[i]]
    print(new_manifolds)
    return results, new_manifolds, ind_coords