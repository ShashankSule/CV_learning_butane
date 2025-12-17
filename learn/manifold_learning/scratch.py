# All imports
import uuid
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import diffusion_map as diffusion_map
from sklearn.neighbors import NearestNeighbors
from feature_maps import IdentityLayer, RecenteringLayer, GramMatrixLayer, \
                        RecenterBondLayer, OrthogonalChangeOfBasisBatched
from _compute_normals import compute_pointcloud_normals
from _dnet_datasets import load_training_configs
from _dnet_architectures import standard_4_layer_dnet_tanh_encoder, standard_4_layer_dnet_tanh_decoder
from _dnet_datasets import DnetData, dnet_dataloader
from _dnet_datasets import save_autoencoder
from _dnet_loss import MatchingLoss, LaplacianLoss

# get the config 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using {device} device")

config_path = '/export/ssule25/CV_learning_butane/learn/manifold_learning/align_experiment_dnet_autoencoder/model_20251216_154905/'
training_configs = load_training_configs(config_path+f'training_configs.yaml')

# get the dmap and get ground truth normals 

dmap_path = training_configs['datapath']
dmap_data = np.load(dmap_path, allow_pickle=True)

feature_data = dmap_data['feature_data']
diff_map = dmap_data['diff_map']
reference_CV = dmap_data['reference_CV']
laplacian = dmap_data['laplacian']
eigvals = dmap_data['eigvals']
# normals, _ = compute_pointcloud_normals(diff_map, method='2d', k_neighbors=100)
# normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
# normals = torch.Tensor(normals).to(device)
breakpoint()
# load the autoencoder
# now set up the architecture
dnet_encoder = standard_4_layer_dnet_tanh_encoder(input_dim=training_configs['input_dim'], encoder_dim=training_configs['encoder_dim']).to(device)
dnet_decoder = standard_4_layer_dnet_tanh_decoder(input_dim=training_configs['input_dim'], encoder_dim=training_configs['encoder_dim']).to(device)

# load the autoencoder
dnet_encoder.load_state_dict(torch.load(config_path + 'model_encoder_state_dict.pth', map_location=device))
dnet_decoder.load_state_dict(torch.load(config_path + 'model_decoder_state_dict.pth', map_location=device))

breakpoint()
# set to eval mode
dnet_encoder.eval()
dnet_outputs = dnet_encoder(torch.Tensor(feature_data).to(device).float()).requires_grad_(True)
cvs = torch.arctan2(dnet_outputs[...,1], dnet_outputs[...,0])
cv_gradients = torch.autograd.grad(cvs.sum(), dnet_outputs, retain_graph=True)[0]
cv_gradients = cv_gradients.cpu().detach().numpy()
# normalize the cv gradients
cv_gradients = cv_gradients / np.linalg.norm(cv_gradients, axis=1, keepdims=True)
plt.figure()
dnet_outputs = dnet_outputs.cpu().detach().numpy()
plt.scatter(dnet_outputs[::100,0], dnet_outputs[::100,1], c=reference_CV[::100], cmap='hsv', alpha=0.5, s=1.0)
# plt.quiver(diff_map[::100,0], diff_map[::100,1], cv_gradients[::100,0], cv_gradients[::100,1], scale=100, scale_units='xy')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('CV Gradients', fontsize=14)
plt.savefig('cv_gradients.png')
plt.close()