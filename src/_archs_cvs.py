import torch.nn as nn
import torch
import sys
import os

# Import the surrogate potential architecture
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from _archs_surrogate_potential import periodic_activation, PsiNetwork


class CVNetwork(nn.Module):
    """
    Collective Variable Network with same architecture as surrogate potential.
    
    This network learns collective variables that are orthogonal to the residence manifold.
    Uses the same architecture as PsiNetwork from _archs_surrogate_potential.py
    """
    
    def __init__(self, activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, output_dim):
        super(CVNetwork, self).__init__()
        
        # Same layers as PsiNetwork
        self.activation = activation
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.hidden3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.hidden4 = nn.Linear(hidden3_dim, hidden4_dim)
        self.bottleneck = nn.Linear(hidden4_dim, output_dim)

        # Sequential model matching PsiNetwork structure
        self.cv_network = nn.Sequential(
            self.hidden1, self.activation, 
            self.hidden2, self.activation,
            self.hidden3, self.activation, 
            self.hidden4, self.activation,
            self.bottleneck
        )

    def forward(self, x):
        """Forward pass through CV network."""
        return self.cv_network(x)
    
    def compute_gradient(self, x):
        """Compute gradient of CV with respect to input."""
        x = x.clone().detach().requires_grad_(True)
        cv_value = self.forward(x)
        
        if cv_value.dim() > 1:
            cv_value = cv_value.squeeze()
            
        grad = torch.autograd.grad(
            outputs=cv_value,
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return grad


class CVDecoder(nn.Module):
    """
    Decoder network to reconstruct manifold points from CV representations.
    
    This provides the reconstruction capability for the CV learning framework.
    """
    
    def __init__(self, cv_dim, manifold_dim, hidden_dims=[32, 64, 32]):
        super(CVDecoder, self).__init__()
        
        layers = []
        prev_dim = cv_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, manifold_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, cv_representation):
        """Reconstruct manifold point from CV representation."""
        return self.decoder(cv_representation)


def standard_cv_network(input_dim, output_dim=1):
    """
    Create a standard CV network with same architecture as surrogate potential.
    
    Parameters
    ----------
    input_dim : int
        Input dimension (manifold dimension)
    output_dim : int
        Output dimension (number of CVs to learn)
        
    Returns
    -------
    CVNetwork
        Initialized CV network
    """
    # Same parameters as standard_4_layer_potential_net
    hidden1_dim = 30
    hidden2_dim = 45
    hidden3_dim = 32
    hidden4_dim = 32
    activation = periodic_activation()
    
    return CVNetwork(activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, output_dim)


def create_cv_with_decoder(input_dim, cv_dim=1, decoder_hidden_dims=[32, 64, 32]):
    """
    Create CV network with associated decoder.
    
    Parameters
    ----------
    input_dim : int
        Input/manifold dimension
    cv_dim : int
        CV output dimension
    decoder_hidden_dims : list
        Hidden layer dimensions for decoder
        
    Returns
    -------
    tuple
        (cv_network, decoder) tuple
    """
    cv_network = standard_cv_network(input_dim, cv_dim)
    decoder = CVDecoder(cv_dim, input_dim, decoder_hidden_dims)
    
    return cv_network, decoder


class MultiCVNetwork(nn.Module):
    """
    Network that learns multiple orthogonal collective variables simultaneously.
    """
    
    def __init__(self, input_dim, num_cvs=2, shared_layers=True):
        super(MultiCVNetwork, self).__init__()
        
        self.num_cvs = num_cvs
        self.shared_layers = shared_layers
        
        if shared_layers:
            # Shared feature extraction layers
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, 30),
                periodic_activation(),
                nn.Linear(30, 45),
                periodic_activation(),
                nn.Linear(45, 32),
                periodic_activation(),
                nn.Linear(32, 32),
                periodic_activation()
            )
            
            # Separate heads for each CV
            self.cv_heads = nn.ModuleList([
                nn.Linear(32, 1) for _ in range(num_cvs)
            ])
            
        else:
            # Separate networks for each CV
            self.cv_networks = nn.ModuleList([
                standard_cv_network(input_dim, 1) for _ in range(num_cvs)
            ])
    
    def forward(self, x):
        """Forward pass returning all CVs."""
        if self.shared_layers:
            features = self.feature_extractor(x)
            cvs = torch.cat([head(features) for head in self.cv_heads], dim=1)
        else:
            cvs = torch.cat([net(x) for net in self.cv_networks], dim=1)
            
        return cvs
    
    def forward_single(self, x, cv_idx):
        """Forward pass for single CV."""
        if cv_idx >= self.num_cvs:
            raise ValueError(f"cv_idx {cv_idx} >= num_cvs {self.num_cvs}")
            
        if self.shared_layers:
            features = self.feature_extractor(x)
            return self.cv_heads[cv_idx](features)
        else:
            return self.cv_networks[cv_idx](x)
    
    def compute_gradient(self, x, cv_idx):
        """Compute gradient of specific CV."""
        x = x.clone().detach().requires_grad_(True)
        cv_value = self.forward_single(x, cv_idx).squeeze()
        
        grad = torch.autograd.grad(
            outputs=cv_value,
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return grad 