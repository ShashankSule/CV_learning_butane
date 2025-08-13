import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import warnings

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    warnings.warn("Open3D not available. 3D normal computation will not work.")


def compute_normals_2d_pca(points, k_neighbors=10, orient_to_center=True):
    """
    Compute normals for 2D pointcloud using local PCA on k-nearest neighbors.
    
    Parameters:
    -----------
    points : np.ndarray
        2D array of shape (N, 2) containing 2D points
    k_neighbors : int
        Number of nearest neighbors to use for local PCA
    orient_to_center : bool
        If True, orient normals to point away from center of mass
        
    Returns:
    --------
    normals : np.ndarray
        Array of shape (N, 2) containing unit normal vectors
    """
    print(f"Computing 2D normals using local PCA with k_neighbors={k_neighbors}")
    
    if points.shape[1] != 2:
        raise ValueError(f"Expected 2D points, got shape {points.shape}")
    
    n_points = points.shape[0]
    normals = np.zeros_like(points)
    
    # Set up k-nearest neighbors
    # Use k+1 because the first neighbor is the point itself
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_points), algorithm='auto')
    nbrs.fit(points)
    
    # Find neighbors for all points
    distances, indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        # Get neighbors (excluding the point itself)
        neighbor_indices = indices[i][1:]  # Skip first index (self)
        
        if len(neighbor_indices) == 0:
            # Fallback: if no neighbors, use arbitrary normal
            normals[i] = np.array([1.0, 0.0])
            continue
            
        neighbors = points[neighbor_indices]
        
        # Center the neighbors around the current point
        centered_neighbors = neighbors - points[i]
        
        if len(centered_neighbors) == 1:
            # Only one neighbor: normal is perpendicular to the line
            direction = centered_neighbors[0]
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            # 2D normal: rotate 90 degrees
            normal = np.array([-direction[1], direction[0]])
        else:
            # Multiple neighbors: use PCA
            # The normal is the direction of minimal variance (smallest eigenvector)
            try:
                # Compute covariance matrix
                cov_matrix = np.cov(centered_neighbors.T)
                
                # Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # Normal is the eigenvector with smallest eigenvalue
                normal = eigenvectors[:, 0]  # Eigenvalues are in ascending order
                
            except np.linalg.LinAlgError:
                # Fallback if PCA fails
                # Use the perpendicular to the mean direction
                mean_direction = np.mean(centered_neighbors, axis=0)
                mean_direction = mean_direction / (np.linalg.norm(mean_direction) + 1e-8)
                normal = np.array([-mean_direction[1], mean_direction[0]])
        
        # Normalize
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        normals[i] = normal
    
    # Ensure all normals are unit length
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Orient normals consistently with respect to center of mass
    if orient_to_center:
        center_of_mass = np.mean(points, axis=0)
        print(f"Center of mass: {center_of_mass}")
        
        # For each point, ensure normal points away from center of mass
        vectors_from_com = points - center_of_mass
        dot_products = np.sum(normals * vectors_from_com, axis=1)
        normals[dot_products < 0] *= -1
        
        flipped_count = np.sum(dot_products < 0)
        print(f"Flipped {flipped_count} out of {n_points} normals ({flipped_count/n_points*100:.1f}%)")
    
    # Verify all normals are unit length
    final_norms = np.linalg.norm(normals, axis=1)
    print(f"Normal lengths - mean: {np.mean(final_norms):.6f}, std: {np.std(final_norms):.6f}")
    print(f"Computed {normals.shape[0]} 2D normals")
    return normals


def compute_normals_3d_open3d(points, radius=0.1, max_neighbors=30, orient_to_center=True):
    """
    Compute normals for 3D pointcloud using Open3D.
    
    Parameters:
    -----------
    points : np.ndarray
        2D array of shape (N, 3) containing 3D points
    radius : float
        Search radius for normal estimation
    max_neighbors : int
        Maximum number of neighbors to consider
    orient_to_center : bool
        If True, orient normals to point away from center of mass
        
    Returns:
    --------
    normals : np.ndarray
        Array of shape (N, 3) containing unit normal vectors
    pcd : open3d.geometry.PointCloud
        The Open3D point cloud object
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required for 3D normal computation")
    
    print(f"Computing 3D normals using Open3D with radius={radius}, max_neighbors={max_neighbors}")
    
    if points.shape[1] != 3:
        raise ValueError(f"Expected 3D points, got shape {points.shape}")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals using hybrid search
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbors)
    )
    
    # Orient normals consistently with respect to center of mass (using Open3D's method)
    if orient_to_center:
        points_array = np.asarray(pcd.points)
        center_of_mass = np.mean(points_array, axis=0)
        print(f"Center of mass: {center_of_mass}")
        
        # Use Open3D's built-in orientation method
        pcd.orient_normals_towards_camera_location(center_of_mass)
        
        # Extract normals and negate to point away from center (Open3D points towards camera)
        normals = -1.0 * np.asarray(pcd.normals)
    else:
        # Extract normals without orientation
        normals = np.asarray(pcd.normals)
    
    # Ensure all normals are unit length (Open3D should already do this, but let's be explicit)
    # norms = np.linalg.norm(normals, axis=1, keepdims=True)
    # normals = normals / (norms + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Verify all normals are unit length
    final_norms = np.linalg.norm(normals, axis=1)
    print(f"Normal lengths - mean: {np.mean(final_norms):.6f}, std: {np.std(final_norms):.6f}")
    print(f"Computed {normals.shape[0]} 3D normals")
    return normals, pcd


def compute_pointcloud_normals(points, method='auto', orient_to_center=True, **kwargs):
    """
    Compute normals for a pointcloud (2D or 3D).
    
    Parameters:
    -----------
    points : np.ndarray
        2D array of shape (N, D) where D is 2 or 3
    method : str
        'auto', '2d', or '3d'. If 'auto', infers from point dimensions
    orient_to_center : bool
        If True, orient normals to point away from center of mass
    **kwargs : dict
        Additional arguments passed to the specific computation method
        
    For 2D (PCA-based):
        - k_neighbors : int (default 10)
        
    For 3D (Open3D-based):
        - radius : float (default 0.1)
        - max_neighbors : int (default 30)
        
    Returns:
    --------
    normals : np.ndarray
        Array of shape (N, D) containing unit normal vectors
    additional_info : dict or None
        For 3D: returns {'pcd': open3d_point_cloud}
        For 2D: returns None
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    if points.ndim != 2:
        raise ValueError(f"Points must be 2D array, got shape {points.shape}")
    
    n_points, n_dims = points.shape
    print(f"Computing normals for {n_points} points in {n_dims}D space")
    
    # Determine method
    if method == 'auto':
        if n_dims == 2:
            method = '2d'
        elif n_dims == 3:
            method = '3d'
        else:
            raise ValueError(f"Cannot auto-determine method for {n_dims}D points. Use method='2d' or '3d'")
    
    # Extract method-specific parameters
    if method == '2d':
        k_neighbors = kwargs.get('k_neighbors', 10)
        normals = compute_normals_2d_pca(points, k_neighbors=k_neighbors, 
                                       orient_to_center=orient_to_center)
        return normals, None
        
    elif method == '3d':
        radius = kwargs.get('radius', 0.1)
        max_neighbors = kwargs.get('max_neighbors', 30)
        print(f"Computing 3D normals using Open3D with radius={radius}, max_neighbors={max_neighbors}")
        normals, pcd = compute_normals_3d_open3d(points, radius=radius, 
                                               max_neighbors=max_neighbors,
                                               orient_to_center=orient_to_center)
        return normals, {'pcd': pcd}
        
    else:
        raise ValueError(f"Unknown method '{method}'. Use '2d', '3d', or 'auto'")


 