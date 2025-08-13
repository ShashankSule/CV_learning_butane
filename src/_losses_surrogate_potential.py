import torch
import torch.nn as nn
import torch.nn.functional as F

def eikonal_loss(inputs, outputs):
    grad_outputs = torch.ones_like(outputs)
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    grad_norm = gradients.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

def data_loss(predicted_potential, target_potential):
    """
    A simple MSE loss for data fitting.
    Assumes target_potential is zero on the manifold.
    """
    return nn.functional.mse_loss(predicted_potential, target_potential)


def gradient_normal_matching_loss(inputs, outputs, target_normals, loss_type='cosine', normalize_gradients=True):
    """
    Loss function that matches gradients of the surrogate potential with computed pointcloud normals.
    
    Parameters:
    -----------
    inputs : torch.Tensor
        Input points where potential is evaluated, shape (batch_size, dim)
        Must have requires_grad=True
    outputs : torch.Tensor
        Predicted potential values, shape (batch_size, 1)
    target_normals : torch.Tensor
        Computed pointcloud normals, shape (batch_size, dim)
    loss_type : str
        Type of loss: 'cosine', 'mse', or 'combined'
    normalize_gradients : bool
        Whether to normalize gradients before comparison
        
    Returns:
    --------
    torch.Tensor
        Scalar loss value
    """
    # Compute gradients of the potential with respect to inputs
    grad_outputs = torch.ones_like(outputs)
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Normalize gradients if requested
    if normalize_gradients:
        grad_norms = gradients.norm(dim=1, keepdim=True)
        gradients = gradients / (grad_norms + 1e-8)  # Add epsilon to avoid division by zero
    
    # Ensure target normals are normalized (they should be, but let's be safe)
    normal_norms = target_normals.norm(dim=1, keepdim=True)
    target_normals_normalized = target_normals / (normal_norms + 1e-8)
    
    if loss_type == 'cosine':
        # Cosine similarity loss: 1 - |cos(angle)| where angle is between gradient and normal
        # We use absolute value because normals can point in either direction
        cosine_sim = F.cosine_similarity(gradients, target_normals_normalized, dim=1)
        loss = 1.0 - torch.abs(cosine_sim)
        return loss.mean()
        
    elif loss_type == 'mse':
        # MSE loss between gradients and normals
        # Note: This assumes gradients and normals should have the same direction and magnitude
        return F.mse_loss(gradients, target_normals_normalized)
        
    elif loss_type == 'combined':
        # Combined loss: cosine similarity + MSE
        cosine_sim = F.cosine_similarity(gradients, target_normals_normalized, dim=1)
        cosine_loss = 1.0 - torch.abs(cosine_sim)
        mse_loss = F.mse_loss(gradients, target_normals_normalized)
        return cosine_loss.mean() + mse_loss
        
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'cosine', 'mse', or 'combined'")


def directional_gradient_loss(inputs, outputs, target_normals, margin=0.1):
    """
    Loss that ensures gradients point in the same direction as normals (allowing for sign flip).
    Uses a margin-based approach similar to triplet loss.
    
    Parameters:
    -----------
    inputs : torch.Tensor
        Input points, shape (batch_size, dim)
    outputs : torch.Tensor
        Predicted potential values, shape (batch_size, 1)
    target_normals : torch.Tensor
        Computed pointcloud normals, shape (batch_size, dim)
    margin : float
        Margin for the directional loss
        
    Returns:
    --------
    torch.Tensor
        Scalar loss value
    """
    # Compute gradients
    grad_outputs = torch.ones_like(outputs)
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Normalize both gradients and normals
    gradients_normalized = F.normalize(gradients, dim=1)
    normals_normalized = F.normalize(target_normals, dim=1)
    
    # Compute dot product (cosine similarity)
    dot_product = torch.sum(gradients_normalized * normals_normalized, dim=1)
    
    # We want |dot_product| to be close to 1 (parallel or anti-parallel)
    # Loss is max(0, margin - |dot_product|)
    alignment_loss = torch.clamp(margin - torch.abs(dot_product), min=0.0)
    
    return alignment_loss.mean()


def normal_consistency_loss(inputs, outputs, target_normals, lambda_eikonal=1.0, lambda_alignment=1.0):
    """
    Combined loss that enforces both Eikonal equation and normal alignment.
    
    Parameters:
    -----------
    inputs : torch.Tensor
        Input points, shape (batch_size, dim)
    outputs : torch.Tensor
        Predicted potential values, shape (batch_size, 1)
    target_normals : torch.Tensor
        Computed pointcloud normals, shape (batch_size, dim)
    lambda_eikonal : float
        Weight for Eikonal loss (gradient magnitude should be 1)
    lambda_alignment : float
        Weight for normal alignment loss
        
    Returns:
    --------
    torch.Tensor
        Scalar loss value
    """
    # Compute gradients
    grad_outputs = torch.ones_like(outputs)
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Eikonal loss: ||∇u|| = 1
    grad_norm = gradients.norm(dim=1)
    eikonal_loss_val = ((grad_norm - 1.0) ** 2).mean()
    
    # Normal alignment loss: gradients should align with normals
    gradients_normalized = F.normalize(gradients, dim=1)
    normals_normalized = F.normalize(target_normals, dim=1)
    
    # Use cosine similarity
    cosine_sim = F.cosine_similarity(gradients_normalized, normals_normalized, dim=1)
    alignment_loss_val = (1.0 - torch.abs(cosine_sim)).mean()
    
    return lambda_eikonal * eikonal_loss_val + lambda_alignment * alignment_loss_val 