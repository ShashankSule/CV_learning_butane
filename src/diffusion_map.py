"""
Diffusion map class, closely following ``pydiffmap'' library 
by Erik Thiede, Zofia Trstanova and Ralf Banisch, 
Github: https://github.com/DiffusionMapsAcademics/pyDiffMap/blob/master/docs/usage.rst
"""
import numpy as np 
import scipy.sparse as sps 
# from scipy.linalg.lapack import clapack as cla
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance as sp_dist
import scipy.linalg as sp_linalg
from tqdm import tqdm
import helpers

def periodic_restrict(x, boundary):
    """Restricts a vector x to comply with periodic boundary conditions

    Args:
        x ([type]): [description]
        boundary ([type]): [description]

    Returns:
        [type]: [description]
    """

    while (x > 0.5*boundary).any():
        x = np.where(x > 0.5*boundary, x - boundary, x) 
    while (x < -0.5*boundary).any(): 
        x = np.where(x < -0.5*boundary, x + boundary, x) 
    return x

def cholesky_hack(C):
    #Computes the (not necessarily unique) Cholesky decomp. for a symmetric positive SEMI-definite matrix, C = LL.T, returns L
    # NOTE: this is a bit more expensive than regular cholesky, should only be used if input matrix is likely not positive definite but it is semi-definite

    # C = MM^T, M^T = QR ---> MM^T = R^T R, so L = R^T
    M = sp_linalg.sqrtm(C)
    R = np.real(np.linalg.qr(M.T)[1])
    return R.T

class DiffusionMap(object):
    r"""
    Class for computing the diffusion map of a given data set. 
    """

    def __init__(self, alpha=0, epsilon="MAX_MIN", num_evecs=1, pbc_dims=None,
                 n_neigh=None, density=None):
        r""" Initialize diffusion map object with basic hyperparameters."""    

        self.alpha = alpha
        self.epsilon = epsilon
        self.num_evecs = num_evecs
        self.pbc_dims = pbc_dims
        self.n_neigh = n_neigh
        self.density = density
        self.flag = False

    def construct_generator(self, data):
        r""" Construct the generator approximation corresponding to input data
        
        Parameters
        ----------
        data: array (num features, num samples)
        
        """  
        K = self._construct_kernel(data)
        N = K.shape[-1]
        print("done with kernel!")

        if self.density is not None:
            q = self.density
        else:
            q = np.array(K.sum(axis=1)).ravel()
        
        # Make right normalizing vector
        q_alpha = np.power(q, -self.alpha) 
        Q_alpha = sps.spdiags(q_alpha, 0, N, N)
        K_rnorm = K.dot(Q_alpha)
        
        # Make left normalizing vector 
        q = np.array(K_rnorm.sum(axis=1)).ravel()
        q_alpha = np.power(q, -1)
        D_alpha = sps.spdiags(q_alpha, 0, N, N)
        P = D_alpha.dot(K_rnorm)
        
        # Transform Markov Matrix P to get discrete generator L 
        L = (P - sps.eye(N, N))/self.epsilon

        self.L = L
        self.K_rnorm = K_rnorm
        #print("switching L for P")
        return self
    
    def fit(self, data):
        r""" Computes the generator and diffusion map for input data

        Parameters
        ----------
        data: array (num features, num samples)

        """  
        self.construct_generator(data) 
        dmap, evecs, evals = self._construct_diffusion_coords(self.L)

        self.dmap = dmap
        self.evecs = evecs
        self.evals = evals

        return self

    def fit_transform(self, data):
        r""" Fits the data as in fit() method, and returns diffusion map 

        Parameters
        ----------
        data: array (num features, num samples)

        """  
        
        self.fit(data) 

        return self.dmap

    def _construct_kernel(self, data):
        r"""Construct kernel matrix of a given data set

        Takes an input data set with structure num_features x
        num_observations, constructs squared distance
        matrix and gaussian kernel

        Parameters
        ----------
        data: array (num features, num samples)
       
        Returns
        -------
        K : array (num samples, num samples)
          pair-wise kernel matrix K(i, j) is kernel
          applied to data points i, j

        """  
        if not self.flag:
            K = self._compute_knn_sq_dists(data)
        else: 
            K = self.sq_dists

        # Construct kernel from data matrix
        K.data = np.exp(-K.data / (self.epsilon))

        # Symmetrizing the kernel due to KNN making a non-symmetric kernel  
        K = 0.5*(K + K.T)

        self.K = K
        return K
    
    def _construct_renormalized_kernel(self, data):
        r""" Construct the renormalized kernel corresponding to input data
        
        Parameters
        ----------
        data: array (num features, num samples)
        
        """  
        K = self._construct_kernel(data)
        N = K.shape[-1]
        print("done with kernel!")

        if self.density is not None:
            q = self.density
        else:
            q = np.array(K.sum(axis=1)).ravel()
        
        # Make right normalizing vector
        q_alpha = np.power(q, -self.alpha) 
        Q_alpha = sps.spdiags(q_alpha, 0, N, N)
        K_rnorm = K.dot(Q_alpha)
        self.K_rnorm = K_rnorm
        return self

    def max_min_epsilon(self, k_alpha = 0.25):
        if not self.flag:
            assert self.sq_dists is None, "Need to compute sq_dists first"
        else: 
            K = self.sq_dists
        k = int(k_alpha*K.shape[0])
        neigh = NearestNeighbors(n_neighbors=k+1,
                                metric='precomputed')
        neigh.fit(self.sq_dists)
        [neigh_dist, neigh_ind] = neigh.kneighbors(self.sq_dists)
        max_epsilon = np.max(neigh_dist[:, k])
        min_epsilon = np.min(neigh_dist[:, 0])
        return max_epsilon, min_epsilon, k 
    
    def choose_epsilon(self):
        r""" Function for automatically choosing epsilon, work in progress

        Parameters
        ----------
        sq_dists: array (num samples, num samples)
            pair-wise squared distance matrix, entry (i, j) 
            is squared euclidean distance between data points i, j
        
        """
        if self.epsilon == "MAX_MIN":
            self.epsilon, k = self.max_min_epsilon()
            print("choosing min_max epsilon with k=%d" % k) 
        
        # otherwise, keep the epsilon value chosen by the user
        #print("epsilon = %f" % self.epsilon) 
        return self

    def ksums(self, data):
        " Compute the sum of the kernel for a range of epsilon values"
        "input: dmap: diffusion_map object, data: (num_features, num_samples) data to compute the kernel sum"
        # first check if sq_dists is computed
        if self.sq_dists is None:
            self._compute_knn_sq_dists(data)
            print("computed sq_dists!")
        eps_max, eps_min, _ = self.max_min_epsilon()
        eps_range = np.logspace(np.log2(0.25*eps_min), np.log2(4.0*eps_max), num=100, base=2)
        kernel_sums = []
        for i in tqdm(range(100)):
            self.epsilon = eps_range[i]
            K = -(1/self.epsilon)*self.sq_dists
            K = K.expm1() # compute exp(K) - 1
            kernel_sum = np.mean(K) + np.mean(np.ones(K.shape))
            kernel_sums.append(kernel_sum)
        return eps_range, kernel_sums
    # Construct a log-linearly spaced range of 100 points between (0.5 * eps_min) and eps_max

    def max_derivative(self, eps_range, kernel_sums): 
        " Compute the maximum discrete derivative of the log of the kernel sum"
        # Compute the discrete derivative of the log of the array kernel_sums
        log_kernel_sums = np.log2(kernel_sums)
        discrete_derivative = np.diff(log_kernel_sums)

        # Find the entry with the maximum discrete derivative
        max_derivative_index = np.argmax(discrete_derivative)
        max_derivative_value = discrete_derivative[max_derivative_index]
        max_eps = eps_range[max_derivative_index]
        return max_eps, max_derivative_index, max_derivative_value, \
            eps_range, discrete_derivative

    def k_sum_test(self, data):
        " Compute the sum of the kernel for a range of epsilon values"
        "input: dmap: diffusion_map object, data: (num_features, num_samples) to compute the kernel sum"
        eps_range, kernel_sums = self.ksums(data)
        max_eps, max_derivative_index, max_derivative_value, \
            eps_range, discrete_derivative = self.max_derivative(eps_range, kernel_sums)
        self.epsilon = max_eps
        return max_eps, max_derivative_index, max_derivative_value, \
            eps_range, discrete_derivative
    
    def semi_group_vals(self, data):
        if self.sq_dists is None:
            self._compute_knn_sq_dists(data)
            print("computed sq_dists!")
        eps_max, eps_min, _ = self.max_min_epsilon()
        eps_range = np.logspace(np.log2(0.25*eps_min), np.log2(4.0*eps_max), num=100, base=2)
        semi_group_vals = []
        for i in tqdm(range(100)):
            self.epsilon = eps_range[i]
            K = -(1/self.epsilon)*self.sq_dists
            K = K.expm1()+1.0
            semi_group_val = np.linalg.norm(K.multiply(K), K**2)
            semi_group_vals.append(semi_group_val)
        return eps_range, semi_group_vals
    
    def semi_group_test(self, data):
        eps_range, semi_group_vals = self.semi_group_vals(data)
        opt_eps = np.argmin(semi_group_vals)

        self.epsilon = opt_eps
        return opt_eps, eps_range, semi_group_vals


    def _construct_diffusion_coords(self, L):
        r""" Description Here

        Parameters
        ----------
        L : array, (num samples, num samples)
            Diffusion map generator matrix 

        Returns
        -------
        dmap : array, (num features, desired number of evecs)
            ith column is the ith `diffusion coordinate'
        evecs : array,  (num features, desired number of evecs)
            ith column is the ith eigenvector of generator
        evals : list, (1, desired number of evecs)
            ith entry is the ith eigval of the generator
        """    
        # Compute eigvals, eigvecs 
        print("computing eigvec matrix") 
        
        evals, evecs = sps.linalg.eigs(L, self.num_evecs + 1, which='LR')
        idx = evals.argsort()[::-1][1:]     # Ignore first eigval / eigfunc
        evals = np.real(evals[idx])
        evecs = np.real(evecs[:, idx])
        dmap = np.dot(evecs, np.diag(np.sqrt(-1./evals)))

        return dmap, evecs, evals

    def get_stationary_dist(self):
        r""" Returns the stationary distribution for the diffusion map markov chain  

        Returns
        -------
        stationary : array, (num samples, 1)
            The stationary distribution for the diffusion map markov chain  
        """    
        
        # Compute left eigvec corresponding to eigval 0
        eval, stationary = sps.linalg.eigs(self.L.T, 1, which='LR')
        stationary = np.real(stationary[:, 0])
        stationary *= np.sign(stationary[0])
        stationary = stationary / np.sum(stationary)        # normalize to 1

        return stationary

    def _compute_knn_sq_dists(self, data):
        r""" Given dataset data, computes matrix of pairwise squared distances and stores sparsely based on k - nearest neighbors

        Parameters
        ----------
        data : array, (num features, num samples)
            data matrix
    
        Returns
        -------
        knn_sq_dists : csr matrix
                       knn-sparse matrix of squared distances
        """

        ##############
        #OLD block of code. this works, but blows up for high dimensional data
        ############## 
        # Construct matrix of pairwise square distances
        #diffs = data.T[np.newaxis, ...] - data.T[:, np.newaxis, ...]
        #if self.pbc_dims is not None:
        #   # Use input pbc_dimensions for distance calculations
        #   diffs = helpers.periodic_restrict(diffs, self.pbc_dims)
        
        ## Construct nearest neighbors graph, sparsify square distances
        #sq_dists = np.sum(diffs**2, axis=-1)

        sq_dists = sp_dist.pdist(data.T, 'sqeuclidean')
        sq_dists = sp_dist.squareform(sq_dists) 

        if self.n_neigh is None:
            self.n_neigh = data.shape[1] # make dense if no param set
            knn_sq_dists = sps.csr_matrix(sq_dists)
        else:
            neigh = NearestNeighbors(n_neighbors=self.n_neigh,
                                 metric='precomputed')
            neigh.fit(sq_dists)
            self.neigh = neigh
            knn_sq_dists = neigh.kneighbors_graph(sq_dists, mode='distance')
        # Compute epsilon from square distance data
        # self.choose_epsilon(sq_dists)
         
        knn_sq_dists.sort_indices()
        self.sq_dists = knn_sq_dists 
        self.flag = True
        return knn_sq_dists
   
    @staticmethod
    def construct_committor(L, B_bool, C_bool):
        r"""Constructs the committor function w.r.t to product set A, reactant set B, C = domain \ (A U B) using the generator L

        Applies boundary conditions and restricts L to solve 
        solve Lq = 0, with q(A) = 0, q(B) = 1

        Parameters
        ----------

        L : sparse array, num data points x num data points
            generator matrix corresponding to a data set, generally the L
                matrix from diffusion maps
        B_bool : boolean vector
            indicates data indices corresponding to reactant B, same length
                as number of data points
        C_bool : boolean vector
            indicates data indices corresponding to transition region domain
                \ (A U B), same length as number of data points

        Returns
        ---------
        q : vector
            Committor function with respect to sets defined by B_bool, C_bool
        """
        Lcb = L[C_bool, :]
        Lcb = Lcb[:, B_bool]
        Lcc = L[C_bool, :]
        Lcc = Lcc[:, C_bool]

        # Assign boundary conditions for q, then solve L(C,C)q(C) = L(C,B)1
        q = np.zeros(L.shape[1])
        q[B_bool] = 1
        row_sum = np.array(np.sum(Lcb, axis=1)).ravel()
        q[C_bool] = sps.linalg.spsolve(Lcc, -row_sum)
        return q

    def get_epsilon(self):
        return self.epsilon

    def get_kernel(self):
        return self.K

    def get_generator(self):
        return self.L

class TargetMeasureDiffusionMap(object):
    r"""
    Class for computing the diffusion map of a given data set. 
    """

    def __init__(self, epsilon, radius=None, n_neigh=None, neigh_mode='RNN',
                 num_evecs=1, target_measure=None, 
                 remove_isolated=True, pbc_dims=None):
        r""" Initialize diffusion map object with basic hyperparameters."""    
        self.epsilon = epsilon
        self.radius = radius
        self.n_neigh = n_neigh
        self.neigh_mode = neigh_mode
        self.num_evecs = num_evecs
        self.target_measure = target_measure
        self.remove_isolated = remove_isolated
        self.pbc_dims = pbc_dims
        self.flag = False

    def fit_transform(self, data):
        r""" Fits the data as in fit() method, and returns diffusion map 

        Parameters
        ----------
        data: array (num features, num samples)

        """  
        
        self.fit(data) 

        return self.dmap

    def fit(self, data):
        r""" Computes the generator and diffusion map for input data

        Parameters
        ----------
        data: array (num features, num samples)

        """  
        self.construct_generator(data) 
        dmap, evecs, evals = self._construct_diffusion_coords()

        self.dmap = dmap
        self.evecs = evecs
        self.evals = evals

        return self

    def _construct_diffusion_coords(self):
        r""" Computes eigenvectors, eigenvalues, and diffusion coordinates of generator 

        Parameters
        ----------
        Returns
        -------
        dmap : array, (num features, desired number of evecs)
            ith column is the ith `diffusion coordinate'
        evecs : array,  (num features, desired number of evecs)
            ith column is the ith eigenvector of generator
        evals : list, (1, desired number of evecs)
            ith entry is the ith eigval of the generator
        """    
        # Symmetrize the generator 
        d = self.stationary_measure
        N = len(d)
        Dinv_onehalf =  sps.spdiags(np.power(d,-0.5), 0, N, N)
        D_onehalf =  sps.spdiags(np.power(d,0.5), 0, N, N)
        Lsymm = D_onehalf @ self.L @ Dinv_onehalf

        # Compute eigvals, eigvecs 
        evals, evecs = sps.linalg.eigsh(Lsymm, k=self.num_evecs + 1, which='SM')

        # Convert back to L^2 norm-1 eigvecs of L 
        evecs = (Dinv_onehalf.toarray()).dot(evecs)
        evecs /= (np.sum(evecs**2, axis=0))**(0.5)
        
        idx = evals.argsort()[::-1][1:]     # Ignore first eigval / eigfunc
        evals = np.real(evals[idx])
        evecs = np.real(evecs[:, idx])
        dmap = np.dot(evecs, np.diag(np.sqrt(-1./evals)))
        return dmap, evecs, evals

    def construct_generator(self, data):
        r"""
        
        Parameters
        ----------
        Returns
        -------
        """
        K = self._construct_kernel(data)
        N = K.shape[-1]                     # Number of data points
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]

        self.rho = self._compute_kde(data)

        # Use kde as target measure if none provided 
        if self.target_measure is None:
            self.target_measure = self.rho

        # Make sure we are using correct indices of the subgraph
        if len(self.target_measure) > N: 
            self.target_measure = self.target_measure[nonisolated_bool]

        if sps.issparse(K):
            # Right Normalize
            rho_inv = np.power(self.rho,-1)
            sqrt_pi = np.power(self.target_measure, 0.5)
            right_normalizer = (sps.spdiags(rho_inv, 0, N, N) 
                                @ sps.spdiags(sqrt_pi, 0 , N, N))
            K_reweight = K @ right_normalizer

            # Left Normalize
            rowsums = np.array(K_reweight.sum(axis=1)).ravel()
            rowsums_inv = np.power(rowsums, -1) 
            left_normalizer = sps.spdiags(rowsums_inv, 0, N, N)
            P = left_normalizer @ K_reweight

            L = (P - sps.eye(N, N))/self.epsilon

            self.stationary_measure = rowsums * rho_inv * sqrt_pi
            self.right_normalizer = right_normalizer 

        else:
            print("Not a sparse kernel, doing dense matrix calculations")
            rho_inv = np.power(self.rho,-1)
            sqrt_pi = np.power(self.target_measure, 0.5)

            # Right Normalize 
            right_normalizer = np.diag(np.power(self.rho,-1)).dot(np.diag(self.target_measure**(0.5)))
            K_reweight = right_normalizer.dot(K.dot(right_normalizer))

            # Left Normalize
            rowsums = np.array(K_reweight.sum(axis=1)).ravel()
            left_normalizer = np.diag(rowsums**(-1))
            P = left_normalizer.dot(K_reweight) 

            L = (P - np.eye(N))/self.epsilon
            self.stationary_measure = rowsums * rho_inv * sqrt_pi
            self.right_normalizer = right_normalizer 

        self.L = L
        self.K_reweight = K_reweight
        return L

    def _construct_kernel(self, data):
        r"""Construct kernel matrix of a given data set

        Takes an input data set with structure num_features x
        num_observations, constructs squared distance
        matrix and gaussian kernel

        Parameters
        ----------
        data: array (num features, num samples)
       
        Returns
        -------
        K : array (num samples, num samples)
          pair-wise kernel matrix K(i, j) is kernel
          applied to data points i, j

        """  
        if not self.flag:
            sqdists = self._compute_sqdists(data)
        else:
            sqdists = self.sqdists
        sqdists = self._compute_nearest_neigh_graph(sqdists)
        
        if sps.issparse(sqdists):
            K = sqdists.copy()
            K.data = np.exp(-K.data / (2*self.epsilon))

            # symmetrize kernel
            K = K.minimum(K.T)

            # Check sparsity of kernel
            num_entries = K.shape[0]**2
            nonzeros_ratio = K.nnz / (num_entries)
            print(f"Ratio of nonzeros to zeros in kernel matrix: {nonzeros_ratio}")
            if nonzeros_ratio > 0.5:
                # Convert to dense matrix
                print("Not a sparse kernel, converting to dense numpy array")
                #self.dense = True
                K = K.toarray()

        else:
            K = np.exp(-sqdists / (2*self.epsilon))
            
            # symmetrize kernel
            K = np.minimum(K,K.T)

        self.K = K

        return K

    def _compute_nearest_neigh_graph(self, sqdists):
        r""" Given dataset data, computes matrix of pairwise squared distances and stores sparsely based on k - nearest neighbors

        Parameters
        ----------
        data : array, (num features, num samples)
            data matrix
    
        Returns
        -------
        sqdists : csr matrix
                      sparse matrix of squared distances
        """
        if self.neigh_mode == 'KNN':
            print("Computing KNN kernel")
            neigh = NearestNeighbors(n_neighbors = self.n_neigh,
                                     metric='precomputed')
            neigh.fit(sqdists)
            sqdists = neigh.kneighbors_graph(sqdists, mode='distance')
        elif self.neigh_mode == 'RNN':
            print("Computing RNN kernel")
            eps_radius = 3*np.sqrt(self.epsilon)
            if self.radius == None:
                self.radius = eps_radius

            neigh = NearestNeighbors(radius = self.radius,
                                     metric='precomputed')
            neigh.fit(sqdists)
            sqdists = neigh.radius_neighbors_graph(sqdists, mode='distance')
        
        # Find isolated indices
        if self.remove_isolated:
            row_sums = np.array(sqdists.sum(axis=1)).ravel()
            nonisolated_bool =  row_sums > 0
        else:
            nonisolated_bool = True*np.ones(sqdists.shape[0], dtype=bool) 
            print("Not leaving out any nodes") 

        # Remove isolated indices from the graph
        sqdists = sqdists[nonisolated_bool, :]
        sqdists = sqdists[:, nonisolated_bool]
        
        self.nn_sqdists = sqdists 

        # Store subgraph of nodes which we use for the algorithm 
        subgraph = {}
        subgraph["nonisolated_bool"] = nonisolated_bool
        print(f"Number of nodes left after removing isolated: {sqdists.shape[0]}")
        self.subgraph = subgraph
        return sqdists

    def _compute_sqdists(self, data):
        sqdists = np.zeros((data.shape[1], data.shape[1]))
        for i in range(sqdists.shape[0]):
            diffs_row = data[:, i, np.newaxis] - data[:, i:]
            if self.pbc_dims is not None: 
                diffs_row = helpers.periodic_restrict(diffs_row, self.pbc_dims) 
            sqdists[i, i:] = np.sum(diffs_row**2, axis=0)
            sqdists[i, 0:i] = sqdists[0:i, i]
        self.flag = True
        self.sqdists = sqdists
        return sqdists

    def _compute_kde(self, data):
        print("Computing kde")
        d = data.shape[0]
        N = self.K.shape[1]
        kde = np.array(self.K.sum(axis=1)).ravel()
        kde *= (N*(2*np.pi*self.epsilon)**(d/2))**(-1) 
        return kde

    def construct_committor(self, B_bool, C_bool):
        r"""Constructs the committor function w.r.t to product set A, reactant set B, C = domain \ (A U B) using the generator L

        Applies boundary conditions and restricts L to solve 
        solve Lq = 0, with q(A) = 0, q(B) = 1

        Parameters
        ----------

        L : sparse array, num data points x num data points
            generator matrix corresponding to a data set, generally the L
                matrix from diffusion maps
        B_bool : boolean vector
            indicates data indices corresponding to reactant B, same length
                as number of data points
        C_bool : boolean vector
            indicates data indices corresponding to transition region domain
                \ (A U B), same length as number of data points

        Returns
        ---------
        q : vector
            Committor function with respect to sets defined by B_bool, C_bool
        """

        # Restrict B, C to subgraph from radius sparsity
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        C_bool = C_bool[nonisolated_bool]
        B_bool = B_bool[nonisolated_bool]

        L = self.L
        Lcb = L[C_bool, :]
        Lcb = Lcb[:, B_bool]
        Lcc = L[C_bool, :]
        Lcc = Lcc[:, C_bool]

        # Assign boundary conditions for q, then solve L(C,C)q(C) = L(C,B)1
        q = np.zeros(L.shape[1])
        q[B_bool] = 1
        row_sum = np.array(np.sum(Lcb, axis=1)).ravel()

        if sps.issparse(L):
            q[C_bool] = sps.linalg.spsolve(Lcc, -row_sum)
        else:
            q[C_bool] = np.linalg.solve(Lcc, -row_sum)
        return q, self.subgraph

    def construct_MFPT(self, B_bool, C_bool):
        r"""Constructs the mean first passage time w.r.t to set B, C = domain \ (B) using the generator L

        Applies boundary conditions and restricts L to solve 
        solve [Lm](C) = -1 and m(A) = 0 
        Parameters
        ----------

        L : sparse array, num data points x num data points
            generator matrix corresponding to a data set, generally the L
                matrix from diffusion maps
        B_bool : boolean vector
            indicates data indices corresponding to reactant B, same length
                as number of data points
        C_bool : boolean vector
            indicates data indices corresponding to complement
              domain \ (B), same length as number of data points

        Returns
        ---------
        m : vector
            mean first passage time with respect to sets defined by B_bool, C_bool
        """

        # Restrict B, C to subgraph from radius sparsity
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        C_bool = C_bool[nonisolated_bool]
        B_bool = B_bool[nonisolated_bool]

        L = self.L
        Lcc = L[C_bool, :]
        Lcc = Lcc[:, C_bool]

        # Assign boundary conditions for q, then solve L(C,C)q(C) = L(C,B)1
        m = np.zeros(L.shape[1])
        m[B_bool] = 0
        mc = m[C_bool]

        if sps.issparse(L):
            m[C_bool] = sps.linalg.spsolve(Lcc, -np.ones_like(mc))
        else:
            m[C_bool] = np.linalg.solve(Lcc, -np.ones_like(mc))
        return m, self.subgraph

    def get_kernel_reweight_symmetric(self):
        K_reweight = self.right_normalizer @ self.K @ self.right_normalizer
        return K_reweight
    
    def get_generator_symmetric(self):
        N = self.L.shape[0]
        if sps.issparse(self.L):
            Lsymm = sps.spdiags(self.stationary_measure, 0, N, N) @ self.L
        else:
            Lsymm = np.diag(self.stationary_measure) @ self.L
        return Lsymm
    
    def get_stationary_measure(self):
        return self.stationary_measure
 
    def get_epsilon(self):
        return self.epsilon

    def get_kernel(self):
        return self.K
    
    def get_sqdists(self):
        return self.sqdists

    def get_subgraph(self):
        return self.subgraph

    def get_generator(self):
        return self.L
    
    def get_evecs(self):
        return self.evecs
    
    def get_evals(self):
        return self.evals

class TargetMeasureMahalanobisDiffusionMap(TargetMeasureDiffusionMap):
    r""" 
    Class for implementing Mahalonobis diffusion maps, replacing the square distance of usual diffusion maps 
    """

    def __init__(self, epsilon, diffusion_list, radius=None, n_neigh=None, neigh_mode='RNN',
                 num_evecs=1, target_measure=None, 
                 remove_isolated=True, pbc_dims=None, using_pinv=False):
        
        # Initialize diffusion map object with basic hyperparameters
        super().__init__(epsilon=epsilon, radius=radius, n_neigh=n_neigh, neigh_mode=neigh_mode,
                         num_evecs=num_evecs, target_measure=target_measure, 
                         remove_isolated=remove_isolated, pbc_dims=pbc_dims)
        self.diffusion_list = diffusion_list 
        self.using_pinv = using_pinv
    def construct_generator(self, data):
        
        K = self._construct_kernel(data)
        N = K.shape[-1]     # Number of data points
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        pi = np.zeros(N)    # initialize right normalization

        self.rho = self._compute_kde(data)

        # Use kde as target measure if none provided
        if self.target_measure is None:
            print("No target measure provided, doing regular MMAP")
            self.target_measure = self.rho

        # Make sure we are using correct indices of the subgraph
        if len(self.target_measure) > N: 
            self.target_measure = self.target_measure[nonisolated_bool]
        if self.diffusion_list.shape[0] > N: 
            self.diffusion_list = self.diffusion_list[nonisolated_bool, :, :]

        # Use determinant in normalizing if we are doing kde normalization or targetMMAP 
        for n in range(N):
            M = self.diffusion_list[n, :, :]
            pi[n] = self.target_measure[n]*((np.linalg.det(M))**(-1/2))

        if sps.issparse(K):
            # Right Normalize
            rho_inv = np.power(self.rho, -1)
            sqrt_pi = np.power(pi, 0.5)
            right_normalizer = (sps.spdiags(rho_inv, 0, N, N) 
                                  @ sps.spdiags(sqrt_pi, 0 , N, N))
            K_reweight = K @ right_normalizer
           
            # Left Normalize
            rowsums = np.array(K_reweight.sum(axis=1)).ravel()
            rowsums_inv = np.power(rowsums, -1)
            left_normalizer = sps.spdiags(rowsums_inv, 0, N, N)
            P = left_normalizer @ K_reweight

            L = (P - sps.eye(N, N))/self.epsilon
            
            
            self.stationary_measure = rowsums * rho_inv * sqrt_pi
            self.right_normalizer = right_normalizer 

        else:
            print("Doing dense kernel matrix calculations")
            rho_inv = np.power(self.rho, -1)
            sqrt_pi = np.power(pi, 0.5)
            # Right Normalize
            right_normalizer = np.diag(rho_inv * sqrt_pi)
            K_reweight = right_normalizer.dot(K.dot(right_normalizer))

            # Left Normalize
            rowsums = np.array(K_reweight.sum(axis=1)).ravel()
            left_normalizer = np.diag(rowsums**(-1))
            P = left_normalizer.dot(K_reweight) 
            
            L = (P - np.eye(N))/self.epsilon
            self.stationary_measure = rowsums * rho_inv * sqrt_pi
            self.right_normalizer = right_normalizer 

        self.L = L
        self.K_reweight = K_reweight
        return L


    def _compute_sqdists(self, data, metric='mahalanobis'):
        r""" Computes matrix of pairwise mahalanobis squared distances 

        Parameters
        ----------
        data : array, (num features, num samples)
            data matrix
    
        Returns
        -------
        mahal_sq_dists : csr matrix
                       knn-sparse matrix of squared distances

        """

        dim = data.shape[0]
        N = data.shape[1]

        sqdists = np.zeros((N, N))
        if metric == 'mahalanobis':
            print("Computing Mahalanobis distance matrix")
            self._compute_inv_chol_covs(N, dim)
            for n in tqdm(range(N)):
                diffs_row = data[:, n, np.newaxis] - data
                if self.pbc_dims is not None: 
                    diffs_row = helpers.periodic_restrict(diffs_row, self.pbc_dims)
                L_row = self.inv_chol_covs[n, :, :]
                if self.using_pinv:
                    # Ldiffs_row = L_row.dot(diffs_row) # this is to get (x-y)^T M(x)(x-y)
                    # sqdists[n, :] = np.einsum('ij, ij -> j', diffs_row, Ldiffs_row)
                    # inner_prods = np.dot(diffs_row.T, np.dot(self.inv_chol_covs[n,:,:], diffs_row))
                    # sqdists[n, :] = np.diag(inner_prods)
                    Ldiffs_row = L_row.dot(diffs_row) # this is to get (x-y)^T LL^T (x-y) = ||L^T (x-y)||
                    sqdists[n, :] = np.sum(Ldiffs_row**2, axis=0)
                else:
                     Ldiffs_row = L_row.dot(diffs_row) # this is to get (x-y)^T LL^T (x-y) = ||L^T (x-y)||
                     sqdists[n, :] = np.sum(Ldiffs_row**2, axis=0)
            sqdists += sqdists.T
            sqdists *= 0.5
        else: 
            for i in range(sqdists.shape[0]):
                diffs_row = data[:, i, np.newaxis] - data[:, i:]
                if self.pbc_dims is not None: 
                    diffs_row = helpers.periodic_restrict(diffs_row, self.pbc_dims) 
                sqdists[i, i:] = np.sum(diffs_row**2, axis=0)
                sqdists[i, 0:i] = sqdists[0:i, i]
        self.flag = True
        self.sqdists = sqdists
        return sqdists

    def _compute_kde(self, data):
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        diffusion_list_nonisolated = self.diffusion_list[nonisolated_bool, :, :]
        N = self.K.shape[1]
        d = data.shape[0]

        kde = np.array(self.K.sum(axis=1)).ravel()
        kde *= (N*(2*np.pi*self.epsilon)**(d/2))**(-1) 
        det_list = np.zeros(N)
        for n in range(N):
            det_list[n] = np.linalg.det(diffusion_list_nonisolated[n, :, :])**(-1/2)
        kde *= det_list
        return kde

    def _compute_inv_chol_covs(self, N, dim):        
        r""" Compute inverse cholesky factorization of input diffusion matrices

        """
        inv_chol_covs = np.zeros((N, dim, dim))
        if self.diffusion_list is not None:
            if self.using_pinv: 
                print("Using pseudo inverses..")
                # self.inv_chol_covs = np.linalg.pinv(self.diffusion_list, rcond=0.01)
                self.inv_chol_covs = self.compute_pinvs(self.diffusion_list)
            else:
                for n in range(N):
                    chol = self.compute_cholesky(self.diffusion_list[n, :, :], n)
                    inv_chol_covs[n, :, :] = np.linalg.inv(chol)
                    # inv_chol_covs[n,:,:] = np.linalg.inv(self.diffusion_list[n,:,:])
                self.inv_chol_covs = inv_chol_covs
        else: 
            print("Defaulting to regular dmaps, no diffusion matrices provided")
            # Make a list of identity matrices
            self.inv_chol_covs = np.ones((N,1,1)) * np.eye(dim)[np.newaxis, :] 
        return self
        
    @staticmethod
    def compute_cholesky(M, n=-1):
        # Error handling block of code for cholesky decomp
        try:
            chol = np.linalg.cholesky(M)
        except np.linalg.LinAlgError as err:
            if 'positive definite' in str(err):
                print(f"Index {n} covar is NOT positive definite, using cholesky hack")
                chol = helpers.cholesky_hack(M)
            else:
                raise
        return chol

    @staticmethod
    def compute_pinvs(M, threshold=0.01):
        "Compute pseudo inverses of M, decomposing M^(-1) = V(Sigma)^-1/2 (Sigma)^(-1/2)V^T = LL^T"
        X, Y = np.linalg.eigh(M)
        Yh = np.transpose(Y[:,:,::-1], (0, 2, 1))  # shape (n, d, d)
        n, d = X.shape

        # Step 1: Compute row-wise max and threshold each row
        row_max = np.max(X, axis=1, keepdims=True)  # shape (n, 1)
        threshold = 0.01 * row_max  # shape (n, 1)
        
        X_thresholded = np.where(X >= threshold, X, 0.0)
        
        # Step 2: Invert non-zero entries and take square root
        with np.errstate(divide='ignore', invalid='ignore'):
            X_inv_sqrt = np.where(X_thresholded > 0, 1.0 / np.sqrt(X_thresholded), 0.0)
        
        # Step 3: Create diagonal matrices
        X_diag = np.zeros((n, d, d))
        diag_indices = np.arange(d)
        X_diag[:, diag_indices, diag_indices] = X_inv_sqrt
        
        # Step 4: Batch matrix multiplication
        result = np.matmul(X_diag, Yh)
        return result 

class NeumannMap(DiffusionMap): 
    def __init__(self, alpha=0.0, epsilon=1.0, num_evecs=3, marked = False, pbc_dims=None,
                 n_neigh=None, density=None, delta=0.5):
        super().__init__(alpha=alpha, epsilon=epsilon,
                         num_evecs=num_evecs, pbc_dims=pbc_dims,
                         n_neigh=n_neigh)
        # marked = boundary, the NMap will always be computed on the unmarked pixels 
        if ~marked: 
            self.marked = marked
        else:
            self.marked = ~marked
        self.delta = delta
    
    def construct_generator(self,data,subgraph): 
        
        # compute map on unmarked points 
        if ~self.marked: 
            subgraph_inds = subgraph
        else: 
            subgraph_inds = ~subgraph
        
        # construct kernel 
        K = self._construct_kernel(data)
        
        # alpha renorm 
        D_alph_inv_vec = (K@np.ones((K.shape[1],1)).flatten())**(-self.alpha)
        D_alph_inv = sps.diags(D_alph_inv_vec)
        K = D_alph_inv@K@D_alph_inv


        # construct graph laplacian
        D = sps.diags(K@np.ones((K.shape[1],1)).flatten())
        L = D - K
        
        # construct Neumann matrix 
        B = K[~subgraph_inds, :][:, subgraph_inds] # boundary matrix 
        delta_TS_vec = B@np.ones((B.shape[1],1)).flatten()
        delta_TS_vec_inv = (1/delta_TS_vec) # deltaT_S matrix 
        delta_TS_inv = sps.diags(delta_TS_vec_inv)
        L_N = self.delta*L[subgraph_inds, :][:, subgraph_inds] - (1-self.delta)*B.T@delta_TS_inv@B # neumann laplacian 
        T_S = D.tocsr()[subgraph_inds,:][:,subgraph_inds] # degree matrix of subgraph 
        K_N = T_S - L_N # Neumann kernel matrix 
        
        # renormalize the kernel matrix
        one_over_T_S_sqrt_vec = 1/(K_N@np.ones((K_N.shape[1],1)).flatten())**(1/2)
        one_over_T_S_sqrt = sps.diags(one_over_T_S_sqrt_vec)
        renormalized_K_N = one_over_T_S_sqrt@K_N@one_over_T_S_sqrt
        T_S_sqrt_vec = (K_N@np.ones((K_N.shape[1],1)).flatten())**(1/2)
        T_S_sqrt = sps.diags(T_S_sqrt_vec)
        P_N = one_over_T_S_sqrt@renormalized_K_N@T_S_sqrt # transition matrix of reflecting random walk 
        
        generator = (P_N - sps.eye(P_N.shape[0]))/self.epsilon # generator of reflecting walk 
        
        self.L = generator
        
        return self 