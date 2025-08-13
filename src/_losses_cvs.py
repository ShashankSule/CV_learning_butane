import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class OrthogonalityLoss(nn.Module):
    """
    Loss function that penalizes the square of the product between 
    the gradient of the CV and the gradient of the surrogate potential.
    
    This enforces orthogonality to the residence manifold.
    """
    
    def __init__(self, normalize_gradients: bool = True):
        super(OrthogonalityLoss, self).__init__()
        self.normalize_gradients = normalize_gradients
        
    def forward(self, cv_gradient: torch.Tensor, potential_gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonality loss.
        
        Parameters
        ----------
        cv_gradient : torch.Tensor
            Gradient of CV with respect to manifold coordinates
        potential_gradient : torch.Tensor  
            Gradient of surrogate potential with respect to manifold coordinates
            
        Returns
        -------
        torch.Tensor
            Orthogonality loss (scalar)
        """
        if self.normalize_gradients:
            cv_grad_normalized = F.normalize(cv_gradient, dim=-1)
            potential_grad_normalized = F.normalize(potential_gradient, dim=-1)
        else:
            cv_grad_normalized = cv_gradient
            potential_grad_normalized = potential_gradient
            
        # Compute inner product and square it
        inner_product = torch.sum(cv_grad_normalized * potential_grad_normalized, dim=-1)
        orthogonality_loss = torch.mean(inner_product ** 2)
        
        return orthogonality_loss


class MatchingLoss(nn.Module):
    """
    Loss function that aligns the gradient of the CV with a chosen column of the hessian.
    
    This is based on the second_CV implementation from compute_V1.ipynb where 
    the CV gradient is matched to a target direction (hessian column).
    """
    
    def __init__(self, loss_type: str = 'mse', normalize_targets: bool = True):
        super(MatchingLoss, self).__init__()
        self.loss_type = loss_type
        self.normalize_targets = normalize_targets
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'cosine':
            self.criterion = nn.CosineSimilarity(dim=-1)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
            
    def forward(self, cv_gradient: torch.Tensor, target_direction: torch.Tensor) -> torch.Tensor:
        """
        Compute matching loss between CV gradient and target direction.
        
        Parameters
        ----------
        cv_gradient : torch.Tensor
            Gradient of CV with respect to manifold coordinates  
        target_direction : torch.Tensor
            Target direction (e.g., hessian column) to match
            
        Returns
        -------
        torch.Tensor
            Matching loss (scalar)
        """
        if self.normalize_targets:
            cv_grad_normalized = F.normalize(cv_gradient, dim=-1)
            target_normalized = F.normalize(target_direction, dim=-1)
        else:
            cv_grad_normalized = cv_gradient
            target_normalized = target_direction
            
        if self.loss_type == 'mse':
            # MSE between normalized gradients
            matching_loss = self.criterion(cv_grad_normalized, target_normalized)
        elif self.loss_type == 'cosine':
            # Maximize cosine similarity (minimize 1 - cosine_similarity)
            cosine_sim = self.criterion(cv_grad_normalized, target_normalized)
            matching_loss = torch.mean(torch.abs(1 - cosine_sim)**2)
            
        return matching_loss


class EikonalLoss(nn.Module):
    """
    Eikonal equation loss that enforces unit norm on gradients.
    
    This is similar to the eikonal loss in compute_V1.ipynb.
    """
    
    def __init__(self):
        super(EikonalLoss, self).__init__()
        
    def forward(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute eikonal loss ||∇CV||² - 1)².
        
        Parameters
        ----------
        gradient : torch.Tensor
            Gradient tensor
            
        Returns
        -------
        torch.Tensor
            Eikonal loss (scalar)
        """
        gradient_norm_squared = torch.sum(gradient ** 2, dim=-1) 
        eikonal_loss = torch.mean((gradient_norm_squared - 1.0) ** 2)
        return eikonal_loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for decoder that takes CV representations back to manifold space.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        super(ReconstructionLoss, self).__init__()
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Parameters
        ----------
        reconstructed : torch.Tensor
            Reconstructed manifold points from decoder
        original : torch.Tensor
            Original manifold points
            
        Returns
        -------
        torch.Tensor
            Reconstruction loss (scalar)
        """
        return self.criterion(reconstructed, original)


class CVLossFunction(nn.Module):
    """
    Combined loss function for collective variable learning.
    
    Combines orthogonality, matching, eikonal, and reconstruction losses
    with configurable weights.
    """
    
    def __init__(
        self,
        orthogonality_weight: float = 1.0,
        matching_weight: float = 1.0, 
        eikonal_weight: float = 1.0,
        reconstruction_weight: float = 0.1,
        normalize_gradients: bool = True,
        matching_loss_type: str = 'mse',
        reconstruction_loss_type: str = 'mse'
    ):
        super(CVLossFunction, self).__init__()
        
        self.orthogonality_weight = orthogonality_weight
        self.matching_weight = matching_weight
        self.eikonal_weight = eikonal_weight
        self.reconstruction_weight = reconstruction_weight
        
        # Initialize individual loss functions
        self.orthogonality_loss = OrthogonalityLoss(normalize_gradients)
        self.matching_loss = MatchingLoss(matching_loss_type, normalize_gradients)
        self.eikonal_loss = EikonalLoss()
        self.reconstruction_loss = ReconstructionLoss(reconstruction_loss_type)
        
    def forward(
        self,
        cv_gradient: torch.Tensor,
        potential_gradient: torch.Tensor, 
        hessian_column: torch.Tensor,
        reconstructed_points: Optional[torch.Tensor] = None,
        original_points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined CV loss.
        
        Parameters
        ----------
        cv_gradient : torch.Tensor
            Gradient of CV network
        potential_gradient : torch.Tensor
            Gradient of surrogate potential
        hessian_column : torch.Tensor
            Target hessian column for matching
        reconstructed_points : torch.Tensor, optional
            Reconstructed manifold points from decoder
        original_points : torch.Tensor, optional
            Original manifold points
            
        Returns
        -------
        tuple
            (total_loss, loss_dict) where loss_dict contains individual losses
        """
        loss_dict = {}
        
        # Orthogonality loss
        orth_loss = self.orthogonality_loss(cv_gradient, potential_gradient)
        loss_dict['orthogonality'] = orth_loss
        
        # Matching loss 
        match_loss = self.matching_loss(cv_gradient, hessian_column)
        loss_dict['matching'] = match_loss
        
        # Eikonal loss
        eik_loss = self.eikonal_loss(cv_gradient)
        loss_dict['eikonal'] = eik_loss
        
        # Reconstruction loss (if decoder outputs provided)
        if reconstructed_points is not None and original_points is not None:
            recon_loss = self.reconstruction_loss(reconstructed_points, original_points)
            loss_dict['reconstruction'] = recon_loss
        else:
            recon_loss = torch.tensor(0.0, device=cv_gradient.device)
            loss_dict['reconstruction'] = recon_loss
        
        # Combine losses
        total_loss = (
            self.orthogonality_weight * orth_loss +
            self.matching_weight * match_loss +
            self.eikonal_weight * eik_loss + 
            self.reconstruction_weight * recon_loss
        )
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


def create_cv_loss_function(config: dict) -> CVLossFunction:
    """
    Create CV loss function from configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with loss weights and settings
        
    Returns
    -------
    CVLossFunction
        Configured loss function
    """
    return CVLossFunction(
        orthogonality_weight=config.get('orthogonality_weight', 1.0),
        matching_weight=config.get('matching_weight', 1.0),
        eikonal_weight=config.get('eikonal_weight', 1.0),
        reconstruction_weight=config.get('reconstruction_weight', 0.1),
        normalize_gradients=config.get('normalize_gradients', True),
        matching_loss_type=config.get('matching_loss_type', 'mse'),
        reconstruction_loss_type=config.get('reconstruction_loss_type', 'mse')
    )


# Utility functions for computing gradients (similar to compute_V1.ipynb)

def compute_cv_gradient(cv_network: nn.Module, manifold_points: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of CV network with respect to manifold points.
    
    Parameters
    ----------
    cv_network : nn.Module
        CV network
    manifold_points : torch.Tensor
        Points on manifold
        
    Returns
    -------
    torch.Tensor
        CV gradients
    """
    # Ensure manifold_points requires grad for proper gradient flow
    if not manifold_points.requires_grad:
        manifold_points = manifold_points.requires_grad_(True)
    
    # Compute CV values
    cv_values = cv_network(manifold_points)
    
    # Compute gradients using autograd.grad for each sample
    gradients = []
    for i in range(cv_values.shape[0]):
        grad = torch.autograd.grad(
            cv_values[i], manifold_points, 
            retain_graph=True, create_graph=True
        )[0][i:i+1]  # Keep only the gradient for the i-th sample
        gradients.append(grad)
    
    return torch.cat(gradients, dim=0)


def compute_jacobian_efficient(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Efficient jacobian computation similar to compute_V1.ipynb implementation.
    
    Parameters
    ----------
    model : nn.Module
        Model to compute jacobian for
    inputs : torch.Tensor
        Input tensor
        
    Returns  
    -------
    torch.Tensor
        Jacobian tensor
    """
    jacobian = torch.autograd.functional.jacobian(model, inputs, create_graph=True)
    
    # Handle different jacobian shapes
    if jacobian.dim() == 4:  # batch case
        jacobian = torch.permute(torch.diagonal(jacobian, dim1=0, dim2=2), (2, 0, 1)).squeeze()
    elif jacobian.dim() == 3:  # single sample case  
        jacobian = jacobian.squeeze()
        
    return jacobian 