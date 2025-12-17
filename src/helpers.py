import numpy as np
import scipy.linalg as sp_linalg

def cholesky_hack(C):
    #Computes the (not necessarily unique) Cholesky decomp. for a symmetric positive SEMI-definite matrix, C = LL.T, returns L
    # NOTE: this is a bit more expensive than regular cholesky, should only be used if input matrix is likely not positive definite but it is semi-definite

    # C = MM^T, M^T = QR ---> MM^T = R^T R, so L = R^T
    M = sp_linalg.sqrtm(C)
    R = np.real(np.linalg.qr(M.T)[1])
    return R.T

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
    