import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import yaml 
from datetime import datetime

# Function to load training configurations from a YAML file
def load_training_configs(config_path):
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
    return configs

def dnet_collate(batch):
    indices = [item[0] for item in batch]  # Extract indices from the batch
    featurized_data = torch.stack([item[1] for item in batch])  # Stack feature data
    diff_map = torch.stack([item[2] for item in batch])  # Stack diffusion map data
    eigvals = torch.stack([item[4] for item in batch])  # Stack eigenvalues

    # Extract the submatrix of laplacian for the batch indices
    laplacian = batch[0][3]  # Access the full Laplacian matrix from the first item
    laplacian_submatrix = laplacian[np.ix_(indices, indices)]  # Submatrix for batch indices

    return torch.tensor(indices), featurized_data, diff_map, torch.tensor(laplacian_submatrix), eigvals

# Modify the DnetData class to store the full Laplacian matrix
class DnetData(torch.utils.data.Dataset):
    """Basic dataset container

    Parameters
    ----------
    featurized_data : Tensor
        The featurized data.
    diff_map : Tensor
        The diffusion map data.
    laplacian : Tensor
        The Laplacian matrix.
    eigvals : Tensor
        The eigenvalues.
    """
    def __init__(self, featurized_data, diff_map, laplacian, eigvals):
        self.featurized_data = featurized_data
        self.diff_map = diff_map
        self.laplacian = laplacian
        self.eigvals = eigvals

    def __getitem__(self, index):
        # Return the index and other data
        return (
            index,
            torch.tensor(self.featurized_data[index]),
            torch.tensor(self.diff_map[index]),
            self.laplacian,  # Pass the full Laplacian matrix
            torch.tensor(self.eigvals)
        )

    def __len__(self):
        return self.featurized_data.shape[0]
    
def dnet_dataloader(dnet_data, batch_size):
    return DataLoader(dnet_data, batch_size=1024, shuffle=True, collate_fn=dnet_collate)

def save_autoencoder(model_encoder, model_decoder, configs, base_dir="dnets"):
    # Create a timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdirectory = f"model_{timestamp}"
    model_dir = f"{base_dir}/{subdirectory}"
    os.makedirs(model_dir, exist_ok=True)

    # Save the model state dict
    model_filename = f"{model_dir}/model_encoder_state_dict.pth"
    torch.save(model_encoder.state_dict(), model_filename)
    if model_decoder is not None:
        decoder_filename = f"{model_dir}/model_decoder_state_dict.pth"
        torch.save(model_decoder.state_dict(), decoder_filename)
    # Save the training configurations as a YAML file
    config_filename = f"{model_dir}/training_configs.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(configs, f, default_flow_style=False)

    print(f"Model and configs saved to: {model_dir}")