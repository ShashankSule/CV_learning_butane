import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import yaml
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src._surrogate_potential_datasets import DNetPointcloudDataset
from src._archs_surrogate_potential import standard_4_layer_potential_net
from src._losses_surrogate_potential import (
    eikonal_loss, data_loss, gradient_normal_matching_loss, 
    directional_gradient_loss, normal_consistency_loss
)
from src.feature_maps import RecenterBondLayer
try:
    from src.feature_maps import featurizer_carbons
except ImportError:
    # If featurizer_carbons is not available, create a simple wrapper
    class FeaturizerCarbons:
        def __init__(self, feature_map):
            self.feature_map = feature_map
            self.carbon_indices = torch.tensor([3, 6, 9, 13])  # Carbon atom indices

        def __call__(self, x):
            batch_size = x.shape[0]
            num_atoms = x.shape[1] // 3
            
            # Apply the initial feature map (e.g., bond alignment)
            aligned_config = self.feature_map(x).view(batch_size, num_atoms, 3)
            
            # Select carbon atoms
            carbons = torch.index_select(aligned_config, 1, self.carbon_indices.to(x.device))
            
            return carbons.flatten(start_dim=1)
        
        def eval(self):
            if hasattr(self.feature_map, 'eval'):
                self.feature_map.eval()
    
    featurizer_carbons = FeaturizerCarbons
from src._dnet_architectures import (
    standard_4_layer_dnet_snake_encoder, standard_4_layer_dnet_snake_encoder_3D,
    standard_4_layer_dnet_tanh_encoder, standard_4_layer_dnet_tanh_encoder_3D
)

def main(config_path):
    # --- Load Configuration ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Configuration ---
    DEVICE = torch.device(config['training_params']['device'])
    DATA_PATH = config['paths']['data_path']
    DNET_MODEL_PATH = config['paths']['dnet_model_path']
    SAVE_PATH = config['paths']['save_path']
    
    BATCH_SIZE = config['training_params']['batch_size']
    NUM_EPOCHS = config['training_params']['num_epochs']
    LR = config['training_params']['lr']
    
    # Loss weights
    DATA_WEIGHT = config['training_params']['data_weight']
    EIKONAL_WEIGHT = config['training_params']['eikonal_weight']
    GRADIENT_WEIGHT = config['training_params']['gradient_matching_weight']
    
    # Gradient matching parameters
    GRADIENT_LOSS_TYPE = config['training_params']['gradient_loss_type']
    NORMALIZE_GRADIENTS = config['training_params']['normalize_gradients']
    
    # Normal computation parameters
    NORMAL_PARAMS = config['normal_params']

    # --- Load Data ---
    data_npz = np.load(DATA_PATH)
    raw_data = data_npz['data']
    
    # --- Feature Map and D-Net ---
    # bondalign_23 uses atoms [1, 2] for bond alignment (data is already carbon-only)
    recenter_map = RecenterBondLayer(atom_ind=[1, 2], batch_mode=True)
    feature_map = recenter_map  # Don't wrap with featurizer_carbons - data is already carbon-only
    
    # Use tanh encoder to match the working script
    dnet_model_base = standard_4_layer_dnet_tanh_encoder(
        input_dim=config['model_params']['dnet']['input_dim'],
        encoder_dim=config['model_params']['dnet']['encoder_dim']
    )
    dnet_model_base.load_state_dict(torch.load(DNET_MODEL_PATH, map_location=DEVICE))
    
    dnet_model = standard_4_layer_dnet_tanh_encoder_3D(dnet_model_base)
    dnet_model.to(DEVICE)
    dnet_model.eval()

    # --- Datasets and Dataloaders ---
    # Create dataset with normal computation if enabled
    compute_normals = NORMAL_PARAMS['compute_normals']
    normal_params = {k: v for k, v in NORMAL_PARAMS.items() if k != 'compute_normals'}
    
    on_manifold_dataset = DNetPointcloudDataset(
        raw_data, feature_map, dnet_model, 
        compute_normals=compute_normals,
        normal_params=normal_params if compute_normals else None
    )
    on_manifold_loader = DataLoader(on_manifold_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Surrogate Potential Model ---
    input_dim = on_manifold_dataset.point_cloud.shape[1]
    potential_model = standard_4_layer_potential_net(
        input_dim=input_dim,
        output_dim=config['model_params']['potential']['output_dim']
    )
    potential_model.to(DEVICE)
    
    optimizer = optim.Adam(potential_model.parameters(), lr=LR)

    # --- Training Loop ---
    print(f"Training with gradient matching: {compute_normals}")
    print(f"Gradient loss type: {GRADIENT_LOSS_TYPE}")
    print(f"Loss weights - Data: {DATA_WEIGHT}, Eikonal: {EIKONAL_WEIGHT}, Gradient: {GRADIENT_WEIGHT}")
    
    for epoch in range(NUM_EPOCHS):
        potential_model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_eikonal_loss = 0.0
        total_gradient_loss = 0.0
        
        for batch_data in on_manifold_loader:
            optimizer.zero_grad()
            
            # Unpack batch data (depends on whether normals are computed)
            if compute_normals:
                _, on_manifold_points, normals = batch_data
                normals = normals.to(DEVICE)
            else:
                _, on_manifold_points = batch_data
                normals = None
            
            on_manifold_points = on_manifold_points.to(DEVICE)
            on_manifold_points.requires_grad_(True)
            
            predicted_potential = potential_model(on_manifold_points)
            target_potential = torch.zeros_like(predicted_potential)
            
            # Compute individual losses
            d_loss = data_loss(predicted_potential, target_potential)
            e_loss = eikonal_loss(on_manifold_points, predicted_potential)
            
            # Initialize total loss
            loss = DATA_WEIGHT * d_loss + EIKONAL_WEIGHT * e_loss
            
            # Add gradient matching loss if normals are available
            g_loss = 0.0
            if compute_normals and normals is not None:
                if GRADIENT_LOSS_TYPE == "cosine":
                    g_loss = gradient_normal_matching_loss(
                        on_manifold_points, predicted_potential, normals, 
                        loss_type='cosine', normalize_gradients=NORMALIZE_GRADIENTS
                    )
                elif GRADIENT_LOSS_TYPE == "mse":
                    g_loss = gradient_normal_matching_loss(
                        on_manifold_points, predicted_potential, normals, 
                        loss_type='mse', normalize_gradients=NORMALIZE_GRADIENTS
                    )
                elif GRADIENT_LOSS_TYPE == "combined":
                    g_loss = gradient_normal_matching_loss(
                        on_manifold_points, predicted_potential, normals, 
                        loss_type='combined', normalize_gradients=NORMALIZE_GRADIENTS
                    )
                elif GRADIENT_LOSS_TYPE == "directional":
                    g_loss = directional_gradient_loss(
                        on_manifold_points, predicted_potential, normals
                    )
                elif GRADIENT_LOSS_TYPE == "consistency":
                    g_loss = normal_consistency_loss(
                        on_manifold_points, predicted_potential, normals,
                        lambda_eikonal=1.0, lambda_alignment=1.0
                    )
                    # For consistency loss, don't add separate eikonal loss
                    loss = DATA_WEIGHT * d_loss + GRADIENT_WEIGHT * g_loss
                else:
                    raise ValueError(f"Unknown gradient loss type: {GRADIENT_LOSS_TYPE}")
                
                # Add gradient loss (except for consistency which handles eikonal internally)
                if GRADIENT_LOSS_TYPE != "consistency":
                    loss = loss + GRADIENT_WEIGHT * g_loss
            
            loss.backward()
            optimizer.step()
            
            # Accumulate losses for reporting
            total_loss += loss.item()
            total_data_loss += d_loss.item()
            total_eikonal_loss += e_loss.item()
            if compute_normals and normals is not None:
                total_gradient_loss += g_loss.item() if isinstance(g_loss, torch.Tensor) else g_loss
        
        # Print epoch statistics
        avg_total = total_loss / len(on_manifold_loader)
        avg_data = total_data_loss / len(on_manifold_loader)
        avg_eikonal = total_eikonal_loss / len(on_manifold_loader)
        
        if compute_normals:
            avg_gradient = total_gradient_loss / len(on_manifold_loader)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Total: {avg_total:.6f}, "
                  f"Data: {avg_data:.6f}, Eikonal: {avg_eikonal:.6f}, "
                  f"Gradient: {avg_gradient:.6f}")
        else:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Total: {avg_total:.6f}, "
                  f"Data: {avg_data:.6f}, Eikonal: {avg_eikonal:.6f}")

    # --- Save Model ---
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(potential_model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='learn/surrogate_potential/config.yml',
                        help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config) 