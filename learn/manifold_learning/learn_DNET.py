import sys
import os
import yaml  # For reading YAML configuration files

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from _dnet_architectures import *
from _dnet_datasets import *
from _dnet_loss import *
import yaml

verbose = True
# Get training configs
config_path = "training_configs.yaml"  # Replace with the actual path to your YAML file
training_configs = load_training_configs(config_path)
# training_configs['datapath'] = 'dmaps/OrthogonalChangeOfBasisBatched_N_10000_epsilon_5.6.npz'
# training_configs['notes'] = 'LAPCAE with laplacian loss and conformal loss for first two coords'
# training_configs['learning_rate'] = 1e-2
# training_configs['num_epochs'] = 1000
# training_configs['activation'] = 'Tanh'
# get dmap data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dic = np.load(training_configs['datapath'], allow_pickle=True)
feature_data, diff_map, laplacian, eigenvalues, reference_CV = data_dic['feature_data'], \
                                                               data_dic['diff_map'], \
                                                               data_dic['laplacian'], \
                                                               data_dic['eigvals'], \
                                                               data_dic['reference_CV']
# training_configs['input_dim'] = feature_data.shape[1]
# training_configs['encoder_dim'] = diff_map.shape[1]

# # # visualize 
# fig = plt.figure()
# ax = plt.subplot()
# ax.scatter(diff_map[:,0], diff_map[:,1], c=reference_CV, cmap='hsv', alpha=0.2)

# Print the loaded configurations
if verbose:
    print("Loaded Training Configurations:")
    print(training_configs)

# Create the dataset
dnet_data = DnetData(featurized_data=feature_data, diff_map=diff_map, laplacian=laplacian, eigvals=eigenvalues)

# Create the DataLoader with the custom collate function
dnet_data_loader = dnet_dataloader(dnet_data, batch_size=training_configs['batch_size'])
print('DataLoader created with batch size:', training_configs['batch_size'])




# Define the model parameters
if training_configs['activation'] == 'Tanh':
    model_encoder_reg = standard_4_layer_dnet_tanh_encoder(input_dim=training_configs['input_dim'], \
                                                            encoder_dim=training_configs['encoder_dim']).to(device)

    model_decoder_reg = standard_4_layer_dnet_tanh_decoder(input_dim=training_configs['input_dim'], \
                                                            encoder_dim=training_configs['encoder_dim']).to(device)
else: 
    model_encoder_reg = standard_4_layer_dnet_snake_encoder(input_dim=training_configs['input_dim'], \
                                                            encoder_dim=training_configs['encoder_dim']).to(device)

    model_decoder_reg = standard_4_layer_dnet_snake_decoder(input_dim=training_configs['input_dim'], \
                                                            encoder_dim=training_configs['encoder_dim']).to(device)

# if initializing with pretrained model: 
initialize_pretrained = False
if initialize_pretrained:
    model_dir =  'dnets/model_20250424_131903/'
    model_path = model_dir + 'model_encoder_state_dict.pth'
    model_encoder_reg.load_state_dict(torch.load(model_path))
    model_decoder_reg.load_state_dict(torch.load(model_path.replace('encoder', 'decoder')))
    print('Pretrained model loaded')


# Instantiate the loss classes
matching_loss_fn = MatchingLoss()
laplacian_loss_fn = LaplacianLoss()

torch.manual_seed(10)
training_configs['loss_weights']['recon_loss'] = 0.0
training_configs['loss_weights']['dnet_loss'] = 1.0
training_configs['loss_weights']['conformal_loss'] = 0.0
training_configs['loss_weights']['laplacian_loss'] = 0.0
training_configs['batch_size'] = diff_map.shape[0]
training_configs['num_epochs'] = 1000
training_configs['scheduler'] = 'StepLR'
training_configs['gamma'] = 0.8
training_configs['step_size'] = 200

# get optimizer 
optimizers = {'encoder': torch.optim.Adam(model_encoder_reg.parameters(), lr=training_configs['learning_rate'], weight_decay=training_configs['weight_decay']),
              'decoder': torch.optim.Adam(model_decoder_reg.parameters(), lr=training_configs['learning_rate'], weight_decay=training_configs['weight_decay'])}

schedulers = {'encoder': torch.optim.lr_scheduler.StepLR(optimizers['encoder'], step_size=training_configs['step_size'], gamma=training_configs['gamma']),
              'decoder': torch.optim.lr_scheduler.StepLR(optimizers['decoder'], step_size=training_configs['step_size'], gamma=training_configs['gamma'])}
def train_step(model_encoder, model_decoder, optimizers, loss_wts, feature_batch, diff_map_batch, laplacian_batch, eigvals_batch):
    model_encoder.train()
    model_decoder.train()
    optimizers['encoder'].zero_grad()
    optimizers['decoder'].zero_grad()

    # move everything to device
    feature_batch = feature_batch.to(device)
    diff_map_batch = diff_map_batch.to(device)
    laplacian_batch = laplacian_batch.to(device)
    eigvals_batch = eigvals_batch.to(device)

    # get conformal loss
    feature_batch.requires_grad_(True)

    # Forward pass
    outputs = model_encoder(feature_batch).double()
    reconstructions = model_decoder(outputs.float())

    conformal_loss = torch.Tensor([0.0]).to(device)
    for i in range(2):
        for j in range(i+1, 2):
            inner_prod = torch.sum(torch.autograd.grad(outputs[:,i].sum(), feature_batch, create_graph=True)[0] * \
                        torch.autograd.grad(outputs[:,j].sum(), feature_batch, create_graph=True)[0], dim=1)
            conformal_loss += matching_loss_fn(inner_prod, torch.zeros_like(inner_prod))

    # collect losses 
    recon_loss = matching_loss_fn(reconstructions, feature_batch)
    laplace_loss = laplacian_loss_fn(laplacian_batch, eigvals_batch, outputs)
    dnet_loss = matching_loss_fn(diff_map_batch, outputs)

    # matching loss
    loss = loss_wts['recon_loss']*recon_loss + \
           loss_wts['laplacian_loss']*laplace_loss + \
           loss_wts['dnet_loss']*dnet_loss + \
           loss_wts['conformal_loss']*conformal_loss 
    # Backward pass and optimization
    loss.backward(retain_graph=True)
    optimizers['encoder'].step()
    optimizers['decoder'].step()
    return loss, recon_loss, laplace_loss, dnet_loss, conformal_loss

def train(model_encoder, model_decoder, training_configs, data_loader, optimizers, schedulers, num_epochs):
    loss_curve = []
    for epoch in range(num_epochs):
        for idx, (indices, feature_batch, diff_map_batch, laplacian_batch, eigvals_batch) in enumerate(data_loader):
            feature_batch = feature_batch.to(device)
            diff_map_batch = diff_map_batch.to(device)
            laplacian_batch = laplacian_batch.to(device)
            eigvals_batch = eigvals_batch.to(device)

            loss, recon_loss, laplace_loss, dnet_loss, conformal_loss = \
                train_step(model_encoder_reg, model_decoder_reg, optimizers, training_configs['loss_weights'], \
                           feature_batch, diff_map_batch, laplacian_batch, eigvals_batch)
            loss_curve.append(loss.item())
            if idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(data_loader)}], \
                      Loss: {loss.item():.4f}, \
                      Recon loss: {recon_loss.item():.4f}, \
                      Conformal loss: {conformal_loss.item():.4f} \
                      Laplacian loss: {laplace_loss.item():.4f} \
                      Dnet loss: {dnet_loss.item():.4f}')
        # Update the learning rate
        for scheduler in schedulers.values():
            scheduler.step()
    return model_encoder, model_decoder, loss_curve

# Train the model
training = False
if training:
    training_configs['num_epochs'] = 1000
    num_epochs = training_configs['num_epochs']
    model_encoder_reg, model_decoder_reg, loss_curve = train(model_encoder_reg, model_decoder_reg, \
                                                             training_configs, dnet_data_loader, \
                                                             optimizers, schedulers, num_epochs)

# Save usage
saving = False
if saving:
    # Save the model and training configs
    training_configs['Notes'] = 'Simple dnet with reconstruction and dnet loss, all atom data'
    training_configs['feature_map'] = 'OrthogonalChangeOfBasisBatched'
    # training_configs['loss_weights'] = {'recon_loss': 1.0, 'laplacian_loss': 0.5, 'dnet_loss': 0.0, 'conformal_loss': 1.0}
    save_autoencoder(model_encoder_reg, model_decoder_reg, training_configs)


# inference time 
inference = True
if inference:
    model_dir = 'dnets/model_20250507_200144/'
    config_path = model_dir + 'training_configs.yaml'  # Replace with the actual path to your YAML file
    training_configs = load_training_configs(config_path)

    # set up data
    data_dic = np.load(training_configs['datapath'], allow_pickle=True)
    feature_data, diff_map, laplacian, eigenvalues, reference_CV = data_dic['feature_data'], \
                                                                data_dic['diff_map'], \
                                                                data_dic['laplacian'], \
                                                                data_dic['eigvals'], \
                                                                data_dic['reference_CV']
    
    # set up model
    if training_configs['activation'] == 'Tanh':
        model_encoder_reg = standard_4_layer_dnet_tanh_encoder(input_dim=training_configs['input_dim'], \
                                                            encoder_dim=training_configs['encoder_dim']).to(device)
    else: 
        model_encoder_reg = standard_4_layer_dnet_snake_encoder(input_dim=training_configs['input_dim'], \
                                                            encoder_dim=training_configs['encoder_dim']).to(device)
    model_path = model_dir + 'model_encoder_state_dict.pth'
    model_encoder_reg.load_state_dict(torch.load(model_path))
    model_encoder_reg.eval()

    # do infe
    # Get the first batch of data
    idx, feature_batch, diff_map_batch, laplacian_batch, eigvals_batch = next(iter(dnet_data_loader))
    with torch.no_grad():
        # feature_batch = feature_batch.to(device)
        # diff_map_batch = diff_map_batch.to(device)
        # laplacian_batch = laplacian_batch.to(device)
        # eigvals_batch = eigvals_batch.to(device)
        outputs = model_encoder_reg(torch.Tensor(feature_data).to(device)).double().cpu().numpy()
        # plt.hist(np.arctan2(outputs[:, 1], outputs[:,0]), bins=100)
        anti = np.abs(reference_CV - np.pi) < 0.2 
        gauche_1 = np.abs(reference_CV - np.pi/3) < 0.1
        gauche_2 = np.abs(reference_CV - 5*np.pi/3) < 0.1
        gauche = np.logical_or(gauche_1, gauche_2).flatten()
        rest = np.logical_not(np.logical_or(anti, gauche)).flatten()
        print(f'Anti: {np.sum(anti)/reference_CV.shape[0]},\
               Gauche: {np.sum(gauche)/reference_CV.shape[0]}, \
                Rest: {np.sum(rest)/reference_CV.shape[0]}')
        A_min, A_max = np.min(-np.arctan2(outputs[anti, 1], outputs[anti,0])), \
            np.max(-np.arctan2(outputs[anti, 1], outputs[anti,0]))
        B1_min, B1_max = np.min(-np.arctan2(outputs[gauche_1, 1], outputs[gauche_1,0])), \
            np.max(-np.arctan2(outputs[gauche_1, 1], outputs[gauche_1,0]))
        B2_min, B2_max = np.min(-np.arctan2(outputs[gauche_2, 1], outputs[gauche_2,0])), \
            np.max(-np.arctan2(outputs[gauche_2, 1], outputs[gauche_2,0]))
        
        print(f'A_min: {A_min}, A_max: {A_max}, \
                B1_min: {B1_min}, B1_max: {B1_max}, \
                B2_min: {B2_min}, B2_max: {B2_max}')
        # plot the results
        plt.scatter(reference_CV[anti], -np.arctan2(outputs[anti, 1], outputs[anti,0]), s=4.0, c='r', label='Anti')
        plt.scatter(reference_CV[gauche], -np.arctan2(outputs[gauche, 1], outputs[gauche,0]), s=4.0, c='b', label='Gauche')
        plt.scatter(reference_CV[rest], -np.arctan2(outputs[rest, 1], outputs[rest,0]), s=4.0, c='g', alpha=0.4)
        plt.xlabel(r'Dihedral Angle $\theta$', fontsize=20)
        plt.ylabel(r'$\xi(x)$', fontsize=20)
        plt.legend()
        plt.title('PlaneAlign CV correlates with dihedral angle', fontsize=20)

        # plt.savefig(model_dir + 'Dnet2d.png')
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(outputs[:,0], outputs[:,1], outputs[:, 2], c=reference_CV, cmap='hsv', alpha=0.2, s=1.0)
        ax.view_init(30, 100)
        plt.show()
        plt.savefig(model_dir + 'Dnet3d.png')

