import torch
import torch.utils.data
import numpy as np
from typing import Optional, Callable, Tuple


class CVDataset(torch.utils.data.Dataset):
    """
    Dataset for learning collective variables orthogonal to the residence manifold.
    
    This dataset:
    1. Loads a feature map, diffusion net, and surrogate potential
    2. For each sample point, provides:
       - Point on diffusion net manifold after feature mapping
       - Gradient of surrogate potential at that point
       - ith column of the hessian of the surrogate potential
    
    Parameters
    ----------
    raw_data : torch.Tensor
        Raw data points to be mapped to manifold
    feature_map : callable
        Function that maps raw data to feature space
    diffusion_net : torch.nn.Module
        Diffusion net model for manifold mapping
    surrogate_potential : torch.nn.Module
        Surrogate potential model
    hessian_component_idx : int
        Index i for which hessian column to compute (0-indexed)
    device : str, optional
        Device for computation (default: 'cpu')
    """
    
    def __init__(
        self,
        raw_data: torch.Tensor,
        feature_map: Callable[[torch.Tensor], torch.Tensor],
        diffusion_net: torch.nn.Module,
        surrogate_potential: torch.nn.Module,
        hessian_component_idx: int = 0,
        device: str = 'cpu'
    ):
        self.raw_data = raw_data.to(device)
        self.feature_map = feature_map
        self.diffusion_net = diffusion_net.to(device)
        self.surrogate_potential = surrogate_potential.to(device)
        self.hessian_component_idx = hessian_component_idx
        self.device = device
        
        # Set models to evaluation mode for consistency
        self.diffusion_net.eval()
        self.surrogate_potential.eval()
        
        # Pre-compute manifold points for efficiency
        self._precompute_manifold_points()
        
    def _precompute_manifold_points(self):
        """Pre-compute points on manifold via feature map + diffusion net."""
        with torch.no_grad():
            # Apply feature map to raw data
            feature_data = torch.stack([self.feature_map(point) for point in self.raw_data])
            
            # Map through diffusion net to get manifold points
            self.manifold_points = self.diffusion_net(feature_data)
            
        print(f"Pre-computed {len(self.manifold_points)} manifold points")
        
    def _compute_potential_gradient(self, manifold_point: torch.Tensor) -> torch.Tensor:
        """Compute gradient of surrogate potential at manifold point."""
        manifold_point = manifold_point.clone().detach().requires_grad_(True)
        
        potential_value = self.surrogate_potential(manifold_point.unsqueeze(0)).squeeze()
        
        # Compute gradient
        grad = torch.autograd.grad(
            outputs=potential_value,
            inputs=manifold_point,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return grad
        
    def _compute_hessian_column(self, manifold_point: torch.Tensor) -> torch.Tensor:
        """Compute ith column of hessian of surrogate potential."""
        manifold_point = manifold_point.clone().detach().requires_grad_(True)
        
        # First compute gradient
        potential_value = self.surrogate_potential(manifold_point.unsqueeze(0)).squeeze()
        grad = torch.autograd.grad(
            outputs=potential_value,
            inputs=manifold_point,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Then compute gradient of the ith gradient component (ith column of hessian)
        if self.hessian_component_idx >= grad.shape[0]:
            raise ValueError(f"hessian_component_idx {self.hessian_component_idx} >= gradient dimension {grad.shape[0]}")
            
        hessian_column = torch.autograd.grad(
            outputs=grad[self.hessian_component_idx],
            inputs=manifold_point,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return hessian_column
        
    def __len__(self) -> int:
        return len(self.raw_data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Returns
        -------
        manifold_point : torch.Tensor
            Point on diffusion net manifold
        potential_grad : torch.Tensor
            Gradient of surrogate potential at manifold point
        hessian_column : torch.Tensor
            ith column of hessian of surrogate potential
        """
        # Get pre-computed manifold point
        manifold_point = self.manifold_points[idx].clone().detach()
        
        # Compute potential gradient and hessian column, then detach for DataLoader compatibility
        potential_grad = self._compute_potential_gradient(manifold_point).detach()
        hessian_column = self._compute_hessian_column(manifold_point).detach()
        
        return manifold_point, potential_grad, hessian_column


class SimpleCVDataset(torch.utils.data.Dataset):
    """
    Simplified dataset for CV learning when manifold points are already available.
    
    Parameters
    ----------
    manifold_points : torch.Tensor
        Points on the manifold
    surrogate_potential : torch.nn.Module
        Surrogate potential model
    hessian_component_idx : int
        Index for which hessian column to compute
    device : str, optional
        Device for computation
    """
    
    def __init__(
        self,
        manifold_points: torch.Tensor,
        surrogate_potential: torch.nn.Module,
        hessian_component_idx: int = 0,
        device: str = 'cpu'
    ):
        self.manifold_points = manifold_points.to(device)
        self.surrogate_potential = surrogate_potential.to(device)
        self.hessian_component_idx = hessian_component_idx
        self.device = device
        
        self.surrogate_potential.eval()
        
    def _compute_potential_gradient(self, manifold_point: torch.Tensor) -> torch.Tensor:
        """Compute gradient of surrogate potential at manifold point."""
        manifold_point = manifold_point.clone().detach().requires_grad_(True)
        
        potential_value = self.surrogate_potential(manifold_point.unsqueeze(0)).squeeze()
        
        grad = torch.autograd.grad(
            outputs=potential_value,
            inputs=manifold_point,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return grad
        
    def _compute_hessian_column(self, manifold_point: torch.Tensor) -> torch.Tensor:
        """Compute ith column of hessian of surrogate potential."""
        manifold_point = manifold_point.clone().detach().requires_grad_(True)
        
        potential_value = self.surrogate_potential(manifold_point.unsqueeze(0)).squeeze()
        grad = torch.autograd.grad(
            outputs=potential_value,
            inputs=manifold_point,
            create_graph=True,
            retain_graph=True
        )[0]
        
        hessian_column = torch.autograd.grad(
            outputs=grad[self.hessian_component_idx],
            inputs=manifold_point,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return hessian_column
        
    def __len__(self) -> int:
        return len(self.manifold_points)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Returns
        -------
        manifold_point : torch.Tensor
            Point on the manifold
        potential_grad : torch.Tensor
            Gradient of surrogate potential at manifold point
        cv_parallel_vector : torch.Tensor
            Cross product of potential gradient and hessian column - vector parallel to CV direction
        """
        manifold_point = self.manifold_points[idx].clone().detach()
        potential_grad = self._compute_potential_gradient(manifold_point).detach()
        hessian_column = self._compute_hessian_column(manifold_point).detach()
        
        # Compute cross product between potential gradient and hessian column
        # This gives a vector parallel to the CV direction
        cv_parallel_vector = torch.cross(potential_grad, hessian_column, dim=0)
        
        return manifold_point, potential_grad, cv_parallel_vector 