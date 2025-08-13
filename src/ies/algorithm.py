# Author: Sharon Zhang
# at https://github.com/sxzhang25/product-manifold-learning
# Modified: Jesse He

import sys
import time
import numpy as np
import scipy.ndimage as ndimage

from utils import *

import networkx as nx

def factorize(
  data, 
  sigma, 
  n_eigenvectors, 
  n_factors, 
  eig_crit, 
  sim_crit,
  K=0, 
  seed=255, 
  exclude_eigs=None, 
  threshold=False,
  k_neighbors = 7,
  self_tuning = False,
  ind_eigs=None,
  verbose=False):
  '''
  an algorithm to factorize a product manifold using selected coordinates

  data: an n x d numpy array containing the original data, where n is
    the number of samples and d is the dimension of the data
  sigma: the width of the kernel
  n_eigenvectors: the number of eigenvectors to compute
  n_factors: the number of factors
  eig_crit: the threshold for the eigenvalue criterion
  sim_crit: the threshold for the similarity criterion
  K: the voting threshold
  ind_eigs: the selected set of independent coordinates
  self_tuning: whether or not to use self-tuning bandwidth selection
  k_neighbors: number of neighbors for self-tuning
  threshold: whether to threshold edge weights at 4*sigma
  '''
  result = {}
  result['data'] = data
  np.random.seed(seed)

  # part 1: create the data graph and compute eigenvectors, eigenvalues
  if verbose:
    print("\nComputing eigenvectors...")
  t0 = time.perf_counter()
  W = calc_W(data, sigma, self_tuning, k_neighbors, threshold)
  phi, Sigma = calc_vars(data, W, sigma, n_eigenvectors=n_eigenvectors)
  t1 = time.perf_counter()
  if verbose:
    print("  Time: %2.2f seconds" % (t1-t0))

  result['phi'] = phi
  result['Sigma'] = Sigma

  # part 2: searching for reliable triplets (combinations)
  if verbose:
    print("\nComputing combos...")
  t0 = time.perf_counter()
  best_matches, best_sims, all_sims = find_combos(phi, Sigma, n_factors, eig_crit, sim_crit, exclude_eigs=exclude_eigs,ind_eigs=ind_eigs)
  assert best_matches
  assert best_sims

  t1 = time.perf_counter()
  if verbose:
    print("  Time: %2.2f seconds" % (t1-t0))
  result['best_matches'] = best_matches
  result['best_sims'] = best_sims
  result['all_sims'] = all_sims
  if verbose:
    print("  Time: %2.2f seconds" % (t1-t0))

  # part 3: identifying separate factor manifolds by eigenvectors
  if verbose:
    print("\nSplitting eigenvectors...")
  t0 = time.perf_counter()
  labels, C = split_eigenvectors(best_matches, best_sims, n_eigenvectors, K,
                                 n_factors=n_factors, verbose=verbose)
  t1 = time.perf_counter()
  result['C_matrix'] = C
  if verbose:
    print("  Time: %2.2f seconds" % (t1-t0))

  manifolds = []
  for m in range(n_factors):
    manifold = labels[0][np.where(labels[1]==m)[0]]
    manifolds.append(manifold)

  # make sure manifold with first nontrivial eigenvector comes first in list
  for idx,m in enumerate(manifolds):
    if 1 in m:
      m1 = manifolds.pop(idx)
      manifolds.insert(0, m1)
  result['manifolds'] = manifolds

  print("\nManifolds...")
  for i,m in enumerate(manifolds):
    print("Manifold #{}".format(i + 1), m)

  return result
