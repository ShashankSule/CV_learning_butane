import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import copy
import pandas as pd

import numpy as np


class Encoder(nn.Module):
    def __init__(self, feature_map, activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim):
        super(Encoder, self).__init__()
        
        # Defining the layers of the neural network
        self.featurizer = feature_map
        self.activation = activation
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.hidden3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.hidden4 = nn.Linear(hidden3_dim, hidden4_dim)
        self.bottleneck = nn.Linear(hidden4_dim, encoder_dim)

        # Collecting layers for convenience
        self.encoder = nn.Sequential(self.hidden1, self.activation, \
                                     self.hidden2, self.activation, \
                                        self.hidden3, self.activation, \
                                            self.hidden4, self.activation, \
                                                self.bottleneck, self.activation)

    
    def encode(self, x):
        y = self.featurizer(x)
        return self.encoder(y)
    
    # Required for any subclass of nn.module: defines how data passes through the `computational graph'
    def forward(self, x):
        # x = self.featurizer(x)
        return self.encode(x)
    

class Decoder(nn.Module):
    def __init__(self, activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim):
        super(Decoder, self).__init__()
        
        # Defining the layers of the neural network
        self.activation = activation
        self.hidden4 = nn.Linear(encoder_dim, hidden4_dim)
        self.hidden3 = nn.Linear(hidden4_dim, hidden3_dim)
        self.hidden2 = nn.Linear(hidden3_dim, hidden2_dim)
        self.hidden1 = nn.Linear(hidden2_dim, hidden1_dim)
        self.reconstruct = nn.Linear(hidden1_dim, input_dim)

        # Collecting layers for convenience as encoder and decoder
        self.decoder = nn.Sequential(self.hidden4, self.activation, self.hidden3, self.activation, self.hidden2, self.activation, self.hidden1, self.activation, self.reconstruct)

    
    def decode(self, z):
        return self.decoder(z)
        
    # Required for any subclass of nn.module: defines how data passes through the `computational graph'
    def forward(self, x):
        return self.decode(x)
    
class DNetDataset(torch.utils.data.Dataset):
    """Basic dataset container

    Parameters
    ----------
     data_tensor : (num_features, num_data) Tensor
         
    """    
    def __init__(self, data_tensor, diffmap_tensor, gram_matrices_tensor): # CHANGED 
        self.data_tensor = data_tensor
        self.diffmap_tensor = diffmap_tensor
        self.gram_matrices = gram_matrices_tensor
    def __getitem__(self, index):
        return index, self.data_tensor[index], self.diffmap_tensor[index], self.gram_matrices[index]
    def __len__(self):
        return self.data_tensor.size(0)

def eigloss(Ltorch, eigvals_mat, encoded_data, indices):
    return torch.mean((torch.matmul(Ltorch[:,indices], encoded_data)[indices,...]-\
                     torch.matmul(encoded_data, eigvals_mat))**2)

def train_sep(model_enc, model_dec, optimizer_enc, optimizer_dec, device, Ltorch, eigvals_mat, train_loader, loss_function, verbose=False): # training for regular autoencoder
    model_enc.train()
    model_dec.train()
    loss1 = 0 # loss associated with the encoder
    loss2 = 0 
    loss3 = 0 
    loss4 = 0 # loss associated with the decoder
    
    # training loop
    for _, (indices, _, diff_map, features) in enumerate(train_loader):
        # data = data.to(device)
        diff_map = diff_map.to(device)
        features = features.to(device)
        # data.requires_grad_(True)
        encoded_data = model_enc(features.float()) 
        decoded_data = model_dec(encoded_data)

        optimizer_enc.zero_grad() 
        optimizer_dec.zero_grad()
        
        loss_tmd = loss_function(encoded_data, diff_map.float())
        loss_dec = loss_function(decoded_data, features.float())
        loss_eigs = eigloss(Ltorch, eigvals_mat, encoded_data, indices)

        loss_enc = loss_tmd + 0.5*loss_eigs

        loss_total = loss_dec + loss_enc
        # loss_enc.backward(retain_graph=True)
        # loss_dec.backward()
        loss_total.backward(retain_graph = True)
        optimizer_enc.step()
        optimizer_dec.step()
        loss1 += loss_tmd.item()
        loss2 += loss_eigs.item()
        loss3 += loss_enc.item()
        loss4 += loss_dec.item()
    if verbose:
        print(f'====> Average loss: {loss1:.4f}, {loss2:.4f}, {loss3:.4f}')
        print('====> Average decoder loss: {:.4f}'.format(
            loss4 / len(train_loader.dataset)))