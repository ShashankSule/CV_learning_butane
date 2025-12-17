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

config_path = "training_configs_align_experiment.yaml"
training_configs = load_training_configs(config_path)

# get the dmap and get ground truth normals 

dmap_path = training_configs['datapath']
dmap_data = np.load(dmap_path, allow_pickle=True)

feature_data = dmap_data['feature_data']
diff_map = dmap_data['diff_map']
reference_CV = dmap_data['reference_CV']
laplacian = dmap_data['laplacian']
eigvals = dmap_data['eigvals']
normals, _ = compute_pointcloud_normals(diff_map, method='2d', k_neighbors=100)
normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
normals = torch.Tensor(normals).to(device)


# set up the dataloader
dnet_dataset = DnetData(featurized_data=feature_data, diff_map=diff_map, laplacian=laplacian, eigvals=eigvals)
dnet_data_loader = dnet_dataloader(dnet_dataset, batch_size=training_configs['batch_size'])
breakpoint()
# now set up the architecture
dnet_encoder = standard_4_layer_dnet_tanh_encoder(input_dim=training_configs['input_dim'], encoder_dim=training_configs['encoder_dim']).to(device)
dnet_decoder = standard_4_layer_dnet_tanh_decoder(input_dim=training_configs['input_dim'], encoder_dim=training_configs['encoder_dim']).to(device)
init_sum_enc = sum([torch.sum(p).item() for p in dnet_encoder.parameters() if p.requires_grad])
init_sum_dec = sum([torch.sum(p).item() for p in dnet_decoder.parameters() if p.requires_grad])
breakpoint()
# get optimizer 
optimizers = {'encoder': torch.optim.Adam(dnet_encoder.parameters(), lr=training_configs['learning_rate'], weight_decay=training_configs['weight_decay'], betas=(0.9, 0.9)),
              'decoder': torch.optim.Adam(dnet_decoder.parameters(), lr=training_configs['learning_rate'], weight_decay=training_configs['weight_decay'], betas=(0.9, 0.9))}

schedulers = {'encoder': torch.optim.lr_scheduler.StepLR(optimizers['encoder'], step_size=training_configs['step_size'], gamma=training_configs['gamma']),
              'decoder': torch.optim.lr_scheduler.StepLR(optimizers['decoder'], step_size=training_configs['step_size'], gamma=training_configs['gamma'])}

# get the losses 
breakpoint()
matching_loss_fn = MatchingLoss()
laplacian_loss_fn = LaplacianLoss()
def orthogonality_loss(normals, outputs): 
    loss = torch.mean(torch.abs(torch.sum(normals * outputs, dim=-1))**2)
    return loss

breakpoint()
# set up the training loop
# define training step
def train_step(model_encoder, model_decoder, optimizers, loss_wts, feature_batch, diff_map_batch, laplacian_batch, eigvals_batch, normals_batch):
    model_encoder.train()
    model_decoder.train()
    optimizers['encoder'].zero_grad()
    optimizers['decoder'].zero_grad()

    # move everything to device
    feature_batch = feature_batch.to(device)
    diff_map_batch = diff_map_batch.to(device)
    laplacian_batch = laplacian_batch.to(device)
    eigvals_batch = eigvals_batch.to(device)
    normals_batch = normals_batch.to(device)

    # predict, with gradients enabled
    feature_batch.requires_grad_(True)
    outputs = model_encoder(feature_batch).double()
    reconstructions = model_decoder(outputs.float())

    # get the losses
    matching_loss = matching_loss_fn(outputs, diff_map_batch)
    laplacian_loss = laplacian_loss_fn(laplacian_batch, eigvals_batch, outputs)
    
    cvs = torch.arctan2(outputs[...,1], outputs[...,0])
    # breakpoint()

    cv_gradients = torch.autograd.grad(cvs.sum(), outputs, retain_graph=True)[0]
    orthogonality_loss_value = orthogonality_loss(normals_batch, cv_gradients)
    reconstruction_loss = matching_loss_fn(reconstructions, feature_batch)

    # collect losses
    loss = loss_wts['dnet_loss']*matching_loss + \
           loss_wts['laplacian_loss']*laplacian_loss + \
           loss_wts['recon_loss']*reconstruction_loss

    # Backward pass and optimization
    loss.backward(retain_graph=True)
    optimizers['encoder'].step()
    optimizers['decoder'].step()

    return loss, matching_loss, laplacian_loss, reconstruction_loss, orthogonality_loss_value

def train(model_encoder, model_decoder, training_configs, data_loader, optimizers, schedulers, num_epochs):
    loss_curve = []
    for epoch in range(num_epochs):
        for idx, (indices, feature_batch, diff_map_batch, laplacian_batch, eigvals_batch) in enumerate(data_loader):
            normals_batch = normals[indices]
            loss, matching_loss, laplacian_loss, reconstruction_loss, orthogonality_loss = train_step(model_encoder, model_decoder, \
                optimizers, training_configs['loss_weights'], \
                    feature_batch, diff_map_batch, laplacian_batch, eigvals_batch, normals_batch)
            loss_curve.append({'loss': loss.item(), 'matching_loss': matching_loss.item(), \
                               'laplacian_loss': laplacian_loss.item(), \
                                'reconstruction_loss': reconstruction_loss.item(), \
                                'orthogonality_loss': orthogonality_loss.item()})
            if idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
        # Update the learning rate
        for scheduler in schedulers.values():
            scheduler.step()
    return model_encoder, model_decoder, loss_curve

# train the model
breakpoint()
training = True
if training:
    dnet_encoder, dnet_decoder, loss_curve = train(dnet_encoder, dnet_decoder, \
                                                   training_configs, dnet_data_loader, optimizers, schedulers,\
                                                   training_configs['num_epochs'])

breakpoint()
final_sum_enc = sum([torch.sum(p).item() for p in dnet_encoder.parameters() if p.requires_grad])
final_sum_dec = sum([torch.sum(p).item() for p in dnet_decoder.parameters() if p.requires_grad])
print(f"Sum of encoder parameters before and after training: {init_sum_enc}, {final_sum_enc}")
print(f"Sum of decoder parameters before and after training: {init_sum_dec}, {final_sum_dec}")

breakpoint()
# generate random key and save loss data
import uuid
key = uuid.uuid4().hex
filename = f'loss_curve_{key}.npy'
np.save(filename, np.array(loss_curve))

# save the model
training_configs['loss_curve_location'] = filename
save_autoencoder(dnet_encoder, dnet_decoder, training_configs, 'align_experiment_dnet_autoencoder')

# 
# # visualize the loss curve 
plt.figure()
plt.plot([item['loss'] for item in loss_curve], label='Total Loss', c='black', linewidth=2)
plt.plot([item['matching_loss'] for item in loss_curve], label='Matching Loss', c='red', linewidth=2)
plt.plot([item['orthogonality_loss'] for item in loss_curve], label='Orthogonality Loss', c='green', linewidth=2)
plt.xlabel('Training Step', fontsize=14)
plt.ylabel('Loss', fontsize=14)
# plt.ylim((0,1.0))
plt.legend()
plt.savefig('loss_curve.png')
plt.close()
# breakpoint()

# visualize cv gradients 
dnet_encoder.eval()
dnet_outputs = dnet_encoder(torch.Tensor(feature_data).to(device).float()).requires_grad_(True)
cvs = torch.arctan2(dnet_outputs[...,1], dnet_outputs[...,0])
cv_gradients = torch.autograd.grad(cvs.sum(), dnet_outputs, retain_graph=True)[0]
cv_gradients = cv_gradients.cpu().detach().numpy()
# normalize the cv gradients
cv_gradients = cv_gradients / np.linalg.norm(cv_gradients, axis=1, keepdims=True)
plt.figure()
dnet_outputs = dnet_outputs.cpu().detach().numpy()
plt.scatter(dnet_outputs[::100,0], dnet_outputs[::100,1], c=reference_CV[::100], cmap='hsv', alpha=0.3, s=1.0)
plt.quiver(diff_map[::100,0], diff_map[::100,1], cv_gradients[::100,0], cv_gradients[::100,1], scale=100, scale_units='xy')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('CV Gradients', fontsize=14)
plt.savefig('cv_gradients.png')
plt.close()
# breakpoint()

