import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import copy
import pandas as pd
from tqdm import tqdm
import numpy as np 
import scipy.sparse as sps
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
import datetime
import argparse
from IPython.display import clear_output

def create_laplacian_sparse(data, target_measure, epsilon, n_neighbors):

    num_features = data.shape[1]
    num_samples = data.shape[0]

    ### Create distance matrix
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='sqeuclidean')
    neigh.fit(data)
    sqdists = neigh.kneighbors_graph(data, mode="distance") 
    print(f"Data type of squared distance matrix: {type(sqdists)}")

    ### Create Kernel
    K = sqdists.copy()
    K.data = np.exp(-K.data / (2*epsilon))
    K = 0.5*(K + K.T)
 
    kde = np.asarray(K.sum(axis=1)).ravel()
    #kde *=  (1.0/num_samples)*(2*np.pi*epsilon)**(-num_features/2) 
    
    # Check sparsity of kernel
    num_entries = K.shape[0]**2
    nonzeros_ratio = K.nnz / (num_entries)
    print(f"Ratio of nonzeros to zeros in kernel matrix: {nonzeros_ratio}")

    ### Create Graph Laplacian
    u = (target_measure**(0.5)) / kde
    U = sps.spdiags(u, 0, num_samples, num_samples) 
    W = U @ K @ U
    stationary = np.asarray(W.sum(axis=1)).ravel()
    inv_stationary = np.power(stationary, -1)
    P = sps.spdiags(inv_stationary, 0, num_samples, num_samples) @ W 
    L = (P - sps.eye(num_samples, num_samples))/epsilon

    return [stationary, K, L]

def create_laplacian_dense(data, target_measure, epsilon):
    num_features = data.shape[1]
    num_samples = data.shape[0]

    ### Create distance matrix
    sqdists = cdist(data, data, 'sqeuclidean') 
    
    ### Create Kernel
    K = np.exp(-sqdists / (2.0*epsilon))

    ### Create Graph Laplacian
    kde = K.sum(axis=1)
    u = (target_measure**(0.5)) / kde
    U = np.diag(u)
    W = U @ K @ U
    stationary = W.sum(axis=1)
    P = np.diag(stationary**(-1)) @ W 
    L = (P - np.eye(num_samples))/epsilon

    return [stationary, K, L]

def compute_spectrum_sparse(L, stationary, num_eigvecs):
    # Symmetrize the generator 
    num_samples = L.shape[0]
    Dinv_onehalf =  sps.spdiags(stationary**(-0.5), 0, num_samples, num_samples)
    D_onehalf =  sps.spdiags(stationary**(0.5), 0, num_samples, num_samples)
    Lsymm = D_onehalf @ L @ Dinv_onehalf

    # Compute eigvals, eigvecs 
    evals, evecs = sps.linalg.eigsh(Lsymm, k=num_eigvecs, which='SM')

    # Convert back to L^2 norm-1 eigvecs of L 
    evecs = (Dinv_onehalf) @ evecs
    evecs /= (np.sum(evecs**2, axis=0))**(0.5)
    
    idx = evals.argsort()[::-1][1:]     # Ignore first eigval / eigfunc
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    dmap = np.dot(evecs, np.diag(np.sqrt(-1./evals)))
    return dmap, evecs, evals

def compute_spectrum_dense(L, stationary, num_eigvecs):
    # Symmetrize the generator 
    Dinv_onehalf =  np.diag(stationary**(-0.5))
    D_onehalf =  np.diag(stationary**(0.5))
    Lsymm = D_onehalf @ L @ Dinv_onehalf

    # Compute eigvals, eigvecs 
    evals, evecs = sps.linalg.eigsh(Lsymm, k=num_eigvecs, which='SM')

    # Convert back to L^2 norm-1 eigvecs of L 
    evecs = (Dinv_onehalf) @ evecs
    evecs /= (np.sum(evecs**2, axis=0))**(0.5)
    
    idx = evals.argsort()[::-1][1:]     # Ignore first eigval / eigfunc
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    return evecs, evals

def compute_gram_matrix_recentered(x):
    # assert x.shape[0]==12
    n_atoms = int(x.shape[0]/3)
    coords = x.reshape((n_atoms,3))
    rescaled_coords = coords - np.mean(coords,axis=0).reshape(1,3)
    gram = np.dot(rescaled_coords, rescaled_coords.T)
    # gram = rescaled_coords
    return gram.flatten()

def epsilon_net(data, ϵ):

    #initialize the net

    dense = True # parameter that checks whether the net is still dense
    # ϵ = 0.005
    iter = 0 
    ϵ_net = np.array(range(data.shape[1]))
    current_point_index = ϵ_net[0]

    #fill the net

    while dense:
        current_point = data[:,current_point_index] # set current point
        ϵ_ball = np.where(np.linalg.norm(data - np.tile(current_point.reshape(current_point.shape[0],1), 
                                                        (1,data.shape[1])), axis=0) <= ϵ)[0] # get indices for ϵ-ball
        ϵ_net = np.delete(ϵ_net, np.where(np.isin(ϵ_net, ϵ_ball))) # kill elements from the ϵ-ball from the net
        ϵ_net = np.append(ϵ_net, current_point_index) # add the current point at the BACK OF THE QUEUE. THIS IS KEY
        current_point_index = ϵ_net[0] # set current point for killing an epsilon ball in the next iteration
        if current_point_index == 0: # if the current point is the initial one, we are done! 
            dense = False
    return ϵ_net, data[:,ϵ_net]
