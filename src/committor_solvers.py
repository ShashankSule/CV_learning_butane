import numpy as np
from scipy.linalg import toeplitz
import scipy.linalg
import scipy.sparse as sps
import scipy.interpolate
import scipy.integrate
import scipy.ndimage


def gaussian_filter_variable(x, sigma, radius_factor=3, mode="reflect"):
    """
    1-D Gaussian smoothing with *variable* bandwidth (σ) and selectable
    boundary handling.

    Parameters
    ----------
    x : (N,) ndarray
        Input signal.
    sigma : float or (N,) ndarray
        Standard deviation(s) of the Gaussian kernel in *samples*.
        Either a scalar (uniform σ) or an array of length N (point-wise σ).
    radius_factor : int, optional
        Kernel is truncated at ±radius_factor * σ (default 3 → ≈99.7 % mass).
    mode : {"reflect", "wrap"}
        Boundary treatment:
          • "reflect" – mirror at the ends (like scipy.ndimage default).  
          • "wrap"    – periodic / circular convolution.

    Returns
    -------
    y : (N,) ndarray
        Smoothed signal, same length as `x`.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")

    if mode not in ("reflect", "wrap"):
        raise ValueError("mode must be 'reflect' or 'wrap'")

    N = x.size
    sigma = np.broadcast_to(sigma, x.shape).astype(float)
    out = np.empty_like(x)

    for i, sig in enumerate(sigma):
        if sig <= 0:  # degenerate σ → copy sample
            out[i] = x[i]
            continue

        r = int(np.ceil(radius_factor * sig))  # half-window radius
        rel_idx = np.arange(-r, r + 1)  # offsets, length 2r+1
        kernel = np.exp(-0.5 * (rel_idx / sig) ** 2)
        kernel /= kernel.sum()

        if mode == "wrap":
            idx = (i + rel_idx) % N  # modular indices
            samples = x[idx]

        else:  # "reflect"
            idx_lo = i - r
            idx_hi = i + r + 1
            # Slice inside bounds
            slice_x = x[max(idx_lo, 0):min(idx_hi, N)]

            # Pad if window overruns an edge
            pad_left = max(0, 0 - idx_lo)
            pad_right = max(0, idx_hi - N)

            if pad_left or pad_right:
                slice_x = np.pad(
                    slice_x,
                    (pad_left, pad_right),
                    mode="reflect"
                )
            samples = slice_x

        out[i] = np.dot(samples, kernel)

    return out


def four(N):
    """Fourier spectral differentiation matrix.
    Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    """
    h = 2 * np.pi / N
    x = h * np.arange(1, N + 1)
    col = np.zeros(N)
    col[1:] = 0.5 * (-1.0) ** np.arange(1, N) / np.tan(np.arange(1, N) * h / 2.0)
    row = np.zeros(N)
    row[0] = col[0]
    row[1:] = col[N - 1:0:-1]
    D = toeplitz(col, row)
    return D, x


def fourdif(y):
    """Apply Fourier differentiation to vector y."""
    N = y.shape[0]
    D, x = four(N)
    return D, x, D @ y


def cheb(N):
    """Chebyshev polynomial differentiation matrix.
    Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    """
    x = -np.cos(np.pi * np.arange(0, N + 1) / N)
    if N % 2 == 0:
        x[N // 2] = 0.0  # only when N is even!
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c = c * (-1.0) ** np.arange(0, N + 1)
    c = c.reshape(N + 1, 1)
    X = np.tile(x.reshape(N + 1, 1), (1, N + 1))
    dX = X - X.T
    D = np.dot(c, 1.0 / c.T) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def chebdif(y):
    """Apply Chebyshev differentiation to vector y."""
    N = y.shape[0]
    D, x = cheb(N)
    return D, x, D @ y


def chebint(x, f):
    """Chebyshev integration assuming x = -np.cos(np.pi*np.arange(0,N+1)/N).
    Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    """
    N = x.shape[0]
    wts = (np.pi / (N)) * (np.sin(np.arange(0, N) * np.pi / (N))) ** 2
    integrand = f / np.sqrt(1 - x ** 2)
    return np.dot(wts, integrand)


class FourierCommittorSolver:
    """Solve committor problem using Fourier spectral methods (for periodic domains)."""
    
    def __init__(self, data, free_energy, Ms, kbT, A_bool, B_bool, C_bool):
        """
        Initialize the Fourier committor solver.
        
        Parameters
        ----------
        data : array_like
            Grid points (assumed periodic)
        free_energy : array_like
            Free energy at grid points
        Ms : array_like
            Diffusion coefficients at grid points
        kbT : float
            Thermal energy
        A_bool : array_like
            Boolean array marking reactant state A
        B_bool : array_like
            Boolean array marking product state B
        C_bool : array_like
            Boolean array marking transition region C
        """
        self.data = np.asarray(data)
        self.free_energy = np.asarray(free_energy)
        self.Ms = np.asarray(Ms)
        self.kbT = kbT
        self.A_bool = A_bool
        self.B_bool = B_bool
        self.C_bool = C_bool
        self.q = None
        self.generator = None
        
    def build_generator(self):
        """Build the generator matrix for the committor equation."""
        N = self.data.shape[0]
        diff_mat, _, forces = fourdif(self.free_energy)
        second_diff_mat = diff_mat @ diff_mat
        
        forces = np.gradient(self.free_energy, self.data)
        self.generator = self.kbT * np.diag(self.Ms) @ second_diff_mat - np.diag(forces) @ diff_mat
        
    def solve_committor(self):
        """Solve the committor function."""
        self.build_generator()
        L = self.generator
        
        # Extract submatrices
        Lcb = L[self.C_bool, :][:, self.B_bool]
        Lcc = L[self.C_bool, :][:, self.C_bool]
        
        # Set boundary conditions
        q = np.zeros(L.shape[1])
        q[self.B_bool] = 1
        
        # Solve for committor in transition region
        row_sum = np.array(np.sum(Lcb, axis=1)).ravel()
        
        if sps.issparse(L):
            q[self.C_bool] = sps.linalg.spsolve(Lcc, -row_sum)
        else:
            q[self.C_bool] = np.linalg.solve(Lcc, -row_sum)
            
        self.q = q
        return q
    
    def compute_rate(self):
        """Compute the transition rate."""
        if self.q is None:
            self.solve_committor()
            
        q = self.q
        _, _, diff_q = fourdif(q)
        
        h = self.data[1] - self.data[0]
        gibbs_measure = np.exp(-self.free_energy / self.kbT)
        Z = h * np.sum(gibbs_measure)
        normalized_gibbs_measure = (1 / Z) * gibbs_measure
        
        rate = self.kbT * h * np.sum(
            self.Ms[self.C_bool] * normalized_gibbs_measure[self.C_bool] * (diff_q[self.C_bool]) ** 2
        )
        return rate


class ChebyshevCommittorSolver:
    """Solve committor problem using Chebyshev spectral methods (for bounded domains)."""
    
    def __init__(self, interior_pts, free_energy_interior, Ms_interior, kbT, full_data=None, full_free_energy=None, N=1000):
        """
        Initialize the Chebyshev committor solver.
        
        Parameters
        ----------
        interior_pts : array_like
            Interior points between boundaries
        free_energy_interior : array_like
            Free energy at interior points
        Ms_interior : array_like
            Diffusion coefficients at interior points
        kbT : float
            Thermal energy
        full_data : array_like, optional
            Full domain data points for computing Z_beta
        full_free_energy : array_like, optional
            Full domain free energy for computing Z_beta
        N : int, optional
            Number of Chebyshev points for interpolation
        """
        self.interior_pts = np.asarray(interior_pts)
        self.free_energy_interior_data = np.asarray(free_energy_interior)
        self.Ms_interior_data = np.asarray(Ms_interior)
        self.kbT = kbT
        self.N = N
        self.a = np.min(interior_pts)
        self.b = np.max(interior_pts)
        
        # Store full domain data for computing Z_beta
        self.full_data = np.asarray(full_data) if full_data is not None else self.interior_pts
        self.full_free_energy = np.asarray(full_free_energy) if full_free_energy is not None else self.free_energy_interior_data
        
    def _build_chebyshev_data(self):
        """Interpolate data to Chebyshev points."""
        D, x = cheb(self.N)
        
        # Transform interior points to [-1, 1] - exactly as in original
        interior_pts_transformed = 2 * (self.interior_pts - self.a) / (self.b - self.a) - 1
        
        # Interpolate free energy and Ms
        free_energy_interpolant = scipy.interpolate.interp1d(
            interior_pts_transformed, self.free_energy_interior_data, 
            kind='linear', fill_value='extrapolate'
        )
        Ms_interpolant = scipy.interpolate.interp1d(
            interior_pts_transformed, self.Ms_interior_data, 
            kind='linear', fill_value='extrapolate'
        )
        
        free_energy_interior = free_energy_interpolant(x)
        Ms_interior = Ms_interpolant(x)
        
        # Compute gradients
        mean_force_interior_data = np.gradient(self.free_energy_interior_data, interior_pts_transformed)
        Ms_grad_interior_data = np.gradient(self.Ms_interior_data, interior_pts_transformed)
        
        mean_force_interpolant = scipy.interpolate.interp1d(
            interior_pts_transformed, mean_force_interior_data, 
            kind='linear', fill_value='extrapolate'
        )
        Ms_grad_interpolant = scipy.interpolate.interp1d(
            interior_pts_transformed, Ms_grad_interior_data, 
            kind='linear', fill_value='extrapolate'
        )
        
        mean_force_interior = mean_force_interpolant(x)
        Ms_grad_interior = Ms_grad_interpolant(x)
        
        return free_energy_interior, Ms_interior, mean_force_interior, Ms_grad_interior, x
    
    def _build_generator(self, free_energy_interior, Ms_interior, mean_force_interior, Ms_grad_interior):
        """Build the generator matrix for Chebyshev BVP."""
        N = free_energy_interior.shape[0] - 1
        D, _ = cheb(N)
        
        # Create matrices
        free_energy_mat = np.tile(free_energy_interior.reshape(-1, 1), (1, N + 1))
        mean_force_mat = np.tile(mean_force_interior.reshape(-1, 1), (1, N + 1))
        Ms_mat = np.tile(Ms_interior.reshape(-1, 1), (1, N + 1))
        Ms_grad_mat = np.tile(Ms_grad_interior.reshape(-1, 1), (1, N + 1))
        
        # Build differential operator
        D1 = self.kbT * Ms_mat * (D @ D)
        D2 = self.kbT * Ms_grad_mat * D - mean_force_mat * Ms_mat * D
        D3 = D1 + D2
        
        rhs = -0.5 * (self.kbT * Ms_grad_interior - mean_force_interior * Ms_interior)
        
        return D3, rhs
    
    def solve_bvp(self):
        """Solve the boundary value problem for committor."""
        # Get data on Chebyshev points
        free_energy_interior, Ms_interior, mean_force_interior, Ms_grad_interior, x = self._build_chebyshev_data()
        
        # Build generator
        D3, rhs = self._build_generator(free_energy_interior, Ms_interior, mean_force_interior, Ms_grad_interior)
        
        # Solve system (with boundary conditions q(-1) = 0, q(1) = 1)
        N = x.shape[0] - 1
        qhat = np.zeros(N + 1)
        qhat[1:-1] = scipy.linalg.solve(D3[1:-1, 1:-1], rhs[1:-1])
        qtilde = qhat + 0.5 * (x + 1)  # Apply boundary conditions
        
        return qtilde, x, Ms_interior, free_energy_interior
    
    def compute_rate(self):
        """Compute the transition rate using Chebyshev quadrature."""
        qtilde, x, Ms_interior, free_energy_interior = self.solve_bvp()
        
        # Get partition function using full domain data
        Z_beta = get_Z_beta(self.full_data, self.full_free_energy, self.kbT)
        
        # Compute rate via Chebyshev quadrature
        N = qtilde.shape[0] - 1
        D, x = cheb(N)
        
        integrand_interpolant = scipy.interpolate.interp1d(
            x, ((D @ qtilde) ** 2) * Ms_interior * np.exp(-free_energy_interior / self.kbT),
            kind='linear', fill_value='extrapolate'
        )
        
        integral_result = scipy.integrate.quad(integrand_interpolant, -1, 1)
        rate = (2 / (self.b - self.a)) * (self.kbT / Z_beta) * integral_result[0]
        
        return rate


def get_Z_beta(data, free_energy, kbT):
    """Compute partition function from data."""
    h = data[1] - data[0]
    Z_beta = h * np.sum(np.exp(-(1 / kbT) * free_energy))
    return Z_beta 