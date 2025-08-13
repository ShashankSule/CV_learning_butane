#!/usr/bin/env python3
"""
Training script for learning collective variables orthogonal to the residence manifold.

This script demonstrates how to use the CV learning framework with all components:
- Datasets for loading manifold data with gradients and hessians
- CV networks with same architecture as surrogate potential  
- Loss functions including orthogonality, matching, eikonal, and reconstruction
- Training loop with proper logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Add src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from _dnet_architectures import standard_4_layer_dnet_tanh_encoder, standard_4_layer_dnet_tanh_encoder_3D

# Add butane_AE path for feature maps
sys.path.append(os.path.join(os.path.dirname(__file__), '../../butane_AE'))
from feature_maps import RecenterBondLayer

# Add path for cv_loader components
sys.path.append(os.path.join(os.path.dirname(__file__), '../../butane_AE/feature_map_align_12_carbons'))

# Define the exact classes from cv_loader.py
class periodic_activation(nn.Module):
    def __init__(self):
        super(periodic_activation, self).__init__()
    def forward(self, x): 
        return x + torch.sin(x)**2

class PsiNetwork(nn.Module):
    def __init__(self, activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim):
        super(PsiNetwork, self).__init__()
        
        # Defining the layers of the neural network
        self.activation = activation
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.hidden3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.hidden4 = nn.Linear(hidden3_dim, hidden4_dim)
        self.bottleneck = nn.Linear(hidden4_dim, encoder_dim)

        # Collecting layers for convenience as encoder and decoder
        self.Psif = nn.Sequential(self.hidden1, self.activation, self.hidden2, self.activation,
                                  self.hidden3, self.activation, self.hidden4, self.activation,
                                  self.bottleneck)
    
    def Psi(self, x):
        return self.Psif(x)
    
    def forward(self, x):
        return self.Psi(x)

from _cv_datasets import SimpleCVDataset
from _archs_cvs import standard_cv_network, create_cv_with_decoder
from _losses_cvs import CVLossFunction, create_cv_loss_function, compute_cv_gradient


class CVTrainer:
    """Trainer class for collective variable learning."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device(self.config['data']['device'] 
                                 if torch.cuda.is_available() 
                                 else 'cpu')
        
        # Set random seed for reproducibility
        torch.manual_seed(self.config['experiment']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['experiment']['seed'])
            
        self._setup_models()
        self._setup_feature_map_and_dnet()
        self._setup_loss_function()
        self._setup_optimizer()
        self._setup_logging()
        
    def _setup_models(self):
        """Set up CV network and decoder."""
        model_config = self.config['model']
        
        # Create CV network
        self.cv_network = standard_cv_network(
            input_dim=model_config['input_dim'],
            output_dim=model_config['cv_output_dim']
        ).to(self.device)
        
        # Create decoder if requested
        if model_config['use_decoder']:
            self.cv_network, self.decoder = create_cv_with_decoder(
                input_dim=model_config['input_dim'],
                cv_dim=model_config['cv_output_dim'],
                decoder_hidden_dims=model_config['decoder_hidden_dims']
            )
            self.cv_network = self.cv_network.to(self.device)
            self.decoder = self.decoder.to(self.device)
        else:
            self.decoder = None
            
        print(f"CV Network: {sum(p.numel() for p in self.cv_network.parameters())} parameters")
        if self.decoder:
            print(f"Decoder: {sum(p.numel() for p in self.decoder.parameters())} parameters")
    
    def _setup_feature_map_and_dnet(self):
        """Set up feature map and diffusion net."""
        model_config = self.config['model']
        
        # Set up feature map
        feature_map_config = model_config['feature_map']
        if feature_map_config['type'] == 'RecenterBondLayer':
            self.feature_map = RecenterBondLayer(
                atom_ind=feature_map_config['atom_indices'], 
                batch_mode=True
            )
            print(f"Feature map: RecenterBondLayer with atoms {feature_map_config['atom_indices']}")
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_config['type']}")
        
        # Set up diffusion net
        dnet_config = model_config['diffusion_net']
        model_path = os.path.join(os.path.dirname(__file__), dnet_config['model_path'])
        
        # Load the base tanh encoder model
        base_model = standard_4_layer_dnet_tanh_encoder(
            input_dim=dnet_config['input_dim'],
            encoder_dim=dnet_config['encoder_dim']
        )
        
        # Load state dict into the base model
        state_dict = torch.load(model_path, map_location=self.device)
        base_model.load_state_dict(state_dict)
        
        # Wrap with the 3D version
        self.diffusion_net = standard_4_layer_dnet_tanh_encoder_3D(base_model)
        self.diffusion_net.to(self.device)
        self.diffusion_net.eval()  # Keep in eval mode
        
        print(f"Diffusion net loaded from: {model_path}")
        print(f"Input dim: {dnet_config['input_dim']}, Encoder dim: {dnet_config['encoder_dim']}")
    
    def _setup_loss_function(self):
        """Set up loss function."""
        self.loss_function = create_cv_loss_function(self.config['loss'])
        
    def _setup_optimizer(self):
        """Set up optimizer and scheduler."""
        training_config = self.config['training']
        
        # Collect parameters from all models
        params = list(self.cv_network.parameters())
        if self.decoder:
            params.extend(list(self.decoder.parameters()))
            
        # Create optimizer
        if training_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        elif training_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                params,
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
            
        # Create scheduler if specified
        scheduler_config = training_config.get('scheduler')
        if scheduler_config and scheduler_config['type']:
            if scheduler_config['type'] == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config['step_size'],
                    gamma=scheduler_config['gamma']
                )
            elif scheduler_config['type'] == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=training_config['epochs']
                )
            elif scheduler_config['type'] == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    patience=scheduler_config['patience'],
                    factor=scheduler_config['gamma']
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
            
    def _setup_logging(self):
        """Set up logging directories."""
        log_config = self.config['logging']
        
        self.log_dir = Path(log_config['log_dir'])
        self.checkpoint_dir = Path(log_config['checkpoint_dir'])
        
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize logging lists
        self.train_losses = []
        self.loss_components = {'orthogonality': [], 'matching': [], 'eikonal': [], 'reconstruction': []}
        
    def load_butane_dataset(self):
        """Load and process butane dataset."""
        print("Loading butane dataset...")
        
        # Load data
        data_config = self.config['data']
        data_path = os.path.join(os.path.dirname(__file__), data_config['data_path'])
        data_file = np.load(data_path)
        
        # Get carbon coordinates
        raw_coords = data_file[data_config['data_key']]  # Shape: (N, 12) for 4 carbons
        print(f"Loaded data shape: {raw_coords.shape}")
        
        # Convert to tensor
        raw_coords_tensor = torch.tensor(raw_coords, dtype=torch.float32)
        
        # Apply feature map (RecenterBondLayer)
        print("Applying feature map...")
        feature_mapped_coords = []
        batch_size = 1000  # Process in batches to avoid memory issues
        
        for i in range(0, len(raw_coords_tensor), batch_size):
            batch = raw_coords_tensor[i:i+batch_size]
            mapped_batch = self.feature_map(batch)
            feature_mapped_coords.append(mapped_batch)
        
        feature_mapped_coords = torch.cat(feature_mapped_coords, dim=0)
        print(f"Feature mapped shape: {feature_mapped_coords.shape}")
        
        # Apply diffusion net mapping
        print("Applying diffusion net mapping...")
        manifold_points = []
        
        with torch.no_grad():  # Don't need gradients for preprocessing
            for i in range(0, len(feature_mapped_coords), batch_size):
                batch = feature_mapped_coords[i:i+batch_size].to(self.device)
                mapped_batch = self.diffusion_net(batch).cpu()
                manifold_points.append(mapped_batch)
        
        manifold_points = torch.cat(manifold_points, dim=0)
        print(f"Final manifold points shape: {manifold_points.shape}")
        
        # Create surrogate potential using EXACTLY the same architecture and loading as get_SDF() in cv_loader.py
        activation_psi = periodic_activation()
        input_dim_psi = 3
        hidden1_dim_psi = 30
        hidden2_dim_psi = 45
        hidden3_dim_psi = 32
        hidden4_dim_psi = 32
        potential_dim_psi = 1
        
        # Create PsiNetwork with exact same architecture
        surrogate_potential = PsiNetwork(activation_psi, input_dim_psi, hidden1_dim_psi, 
                                       hidden2_dim_psi, hidden3_dim_psi, hidden4_dim_psi, potential_dim_psi)
        
        # Load the pretrained model from the same location as cv_loader.py
        potential_model_path = os.path.join(os.path.dirname(__file__), 
                                          '../../butane_AE/feature_map_align_12_carbons/potential_oct8_carbons_01')
        surrogate_potential.load_state_dict(torch.load(potential_model_path, map_location='cpu'))
        
        # Set all parameters to not require gradients (frozen model)
        for param in surrogate_potential.parameters():
            param.requires_grad = False
        
        print(f"Loaded pretrained surrogate potential from: {potential_model_path}")
        
        # Create dataset
        dataset = SimpleCVDataset(
            manifold_points=manifold_points,
            surrogate_potential=surrogate_potential,
            hessian_component_idx=self.config['data']['hessian_component_idx'],
            device=self.device
        )
        
        return dataset
    
    def train_epoch(self, data_loader):
        """Train for one epoch."""
        self.cv_network.train()
        if self.decoder:
            self.decoder.train()
            
        epoch_loss = 0.0
        epoch_components = {key: 0.0 for key in self.loss_components.keys()}
        
        for batch_idx, (manifold_points, potential_gradients, cv_parallel_vectors) in enumerate(data_loader):
            manifold_points = manifold_points.to(self.device)
            potential_gradients = potential_gradients.to(self.device)
            cv_parallel_vectors = cv_parallel_vectors.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Compute CV gradients
            cv_gradients = compute_cv_gradient(self.cv_network, manifold_points)
            
            # Compute reconstruction if decoder is available
            reconstructed_points = None
            if self.decoder:
                cv_values = self.cv_network(manifold_points)
                reconstructed_points = self.decoder(cv_values)
            
            # Compute loss
            total_loss, loss_dict = self.loss_function(
                cv_gradient=cv_gradients,
                potential_gradient=potential_gradients,
                hessian_column=cv_parallel_vectors,  # Now using cross product vector
                reconstructed_points=reconstructed_points,
                original_points=manifold_points if reconstructed_points is not None else None
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping if specified
            grad_clip = self.config['training'].get('grad_clip')
            if grad_clip:
                params = list(self.cv_network.parameters())
                if self.decoder:
                    params.extend(list(self.decoder.parameters()))
                nn.utils.clip_grad_norm_(params, grad_clip)
            
            self.optimizer.step()
            
            # Update statistics
            epoch_loss += total_loss.item()
            for key, value in loss_dict.items():
                if key in epoch_components:
                    epoch_components[key] += value.item()
                    
            # Log batch statistics
            if batch_idx % self.config['logging']['log_interval'] == 0:
                print(f'Batch {batch_idx}/{len(data_loader)}: Loss = {total_loss.item():.6f}')
                
        # Average over batches
        epoch_loss /= len(data_loader)
        for key in epoch_components:
            epoch_components[key] /= len(data_loader)
            
        return epoch_loss, epoch_components
    
    def validate(self, data_loader):
        """Validate the model."""
        self.cv_network.eval()
        if self.decoder:
            self.decoder.eval()
            
        val_loss = 0.0
        val_components = {key: 0.0 for key in self.loss_components.keys()}
        
        with torch.no_grad():
            for manifold_points, potential_gradients, cv_parallel_vectors in data_loader:
                manifold_points = manifold_points.to(self.device)
                potential_gradients = potential_gradients.to(self.device)
                cv_parallel_vectors = cv_parallel_vectors.to(self.device)
                
                # Compute CV gradients
                cv_gradients = compute_cv_gradient(self.cv_network, manifold_points)
                
                # Compute reconstruction if available
                reconstructed_points = None
                if self.decoder:
                    cv_values = self.cv_network(manifold_points)
                    reconstructed_points = self.decoder(cv_values)
                
                # Compute loss
                total_loss, loss_dict = self.loss_function(
                    cv_gradient=cv_gradients,
                    potential_gradient=potential_gradients,
                    hessian_column=cv_parallel_vectors,  # Now using cross product vector
                    reconstructed_points=reconstructed_points,
                    original_points=manifold_points if reconstructed_points is not None else None
                )
                
                val_loss += total_loss.item()
                for key, value in loss_dict.items():
                    if key in val_components:
                        val_components[key] += value.item()
        
        # Average over batches
        val_loss /= len(data_loader)
        for key in val_components:
            val_components[key] /= len(data_loader)
            
        return val_loss, val_components
    
    def train(self):
        """Main training loop."""
        print("Starting CV training...")
        
        # Load real butane dataset
        dataset = self.load_butane_dataset()
        
        # Split into train/validation
        train_size = int(self.config['data']['train_test_split'] * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            
            # Train
            train_loss, train_components = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_components = None, None
            if self.config['validation']['enabled'] and epoch % self.config['validation']['val_interval'] == 0:
                val_loss, val_components = self.validate(val_loader)
                
                print(f"Validation Loss: {val_loss:.6f}")
                
                # Early stopping
                if self.config['training']['early_stopping']['enabled']:
                    if val_loss < best_val_loss - self.config['training']['early_stopping']['min_delta']:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        self.save_checkpoint(epoch, is_best=True)
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.config['training']['early_stopping']['patience']:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log progress
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Components - Orth: {train_components['orthogonality']:.4f}, "
                  f"Match: {train_components['matching']:.4f}, "
                  f"Eikonal: {train_components['eikonal']:.4f}, "  
                  f"Recon: {train_components['reconstruction']:.4f}")
            
            # Save checkpoint
            if epoch % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(epoch)
            
            # Update logging
            self.train_losses.append(train_loss)
            for key, value in train_components.items():
                self.loss_components[key].append(value)
        
        print("Training completed!")
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'cv_network_state_dict': self.cv_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'loss_components': self.loss_components
        }
        
        if self.decoder:
            checkpoint['decoder_state_dict'] = self.decoder.state_dict()
            
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}")
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(self.train_losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Loss components
        axes[0, 1].plot(self.loss_components['orthogonality'], label='Orthogonality')
        axes[0, 1].plot(self.loss_components['matching'], label='Matching')
        axes[0, 1].set_title('Primary Loss Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        axes[1, 0].plot(self.loss_components['eikonal'], label='Eikonal')
        axes[1, 0].plot(self.loss_components['reconstruction'], label='Reconstruction')
        axes[1, 0].set_title('Secondary Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        
        # Learning rate (if scheduler is used)
        if self.scheduler:
            current_lr = self.optimizer.param_groups[0]['lr']
            axes[1, 1].axhline(y=current_lr, color='r', linestyle='--')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Scheduler', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to {self.log_dir / 'training_curves.png'}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train collective variable network')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CVTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.cv_network.load_state_dict(checkpoint['cv_network_state_dict'])
        if trainer.decoder and 'decoder_state_dict' in checkpoint:
            trainer.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main() 