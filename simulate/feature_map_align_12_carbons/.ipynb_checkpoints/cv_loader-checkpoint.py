import torch
from torch import nn
import os
import sys
from butane_AE.feature_maps import * 

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# # define architectures 
# 1. feature map

selected_atoms = [3, 6, 9, 13]  # Example: select the carbons
N = 14
carbons_matrix = torch.zeros(len(selected_atoms), N)
for i, atom in enumerate(selected_atoms):
    carbons_matrix[i, atom] = 1

feature_map = RecenterBondLayer([6, 9], batch_mode=True)

print(f'{script_dir}')
class featurizer_carbons(nn.Module):
    def __init__(self, carbons_matrix, feature_map, indexing=True):
        super(featurizer_carbons, self).__init__()
        self.carbons_matrix = carbons_matrix 
        self.feature_map = feature_map
        self.indexing = indexing
    def forward(self, x): 
    # note that x has to be (batch_size, 42)
        batch_size = x.shape[0]
        aligned_config = self.feature_map(x).reshape((batch_size, 14,3))
        
        # if not self.indexing:
        #     carbons = torch.matmul(self.carbons_matrix, aligned_config)
        # else:
        #     assert aligned_config is not None
        #     carbons = torch.index_select(aligned_config, -2, torch.tensor([3,6,9,13]))
        # return carbons.flatten(start_dim=1)
        
        return aligned_config 

# 2. diffusion net 
class Encoder(nn.Module):
    def __init__(self, activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim):
        super(Encoder, self).__init__()
        
        # Defining the layers of the neural network
        # self.featurizer = feature_map
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
        # y = self.featurizer(x)
        # y = self.encoder(x)
        return self.encoder(x)
    
    # Required for any subclass of nn.module: defines how data passes through the `computational graph'
    def forward(self, x):
        # x = self.featurizer(x)
        return self.encode(x)[...,:-1]

class periodic_activation(nn.Module):
    def __init__(self):
        super(periodic_activation, self).__init__()
    def forward(self,x): 
        return x + torch.sin(x)**2

class PsiNetwork(nn.Module): # learns the potential function V_1 given its zero level set 
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
        self.Psif = nn.Sequential(self.hidden1, self.activation, self.hidden2, self.activation, \
                                  self.hidden3, self.activation, self.hidden4, self.activation, \
                                  self.bottleneck)
        # the sigmoid is to make sure the values are between 0 and 1.
    
    def Psi(self, x):
        return self.Psif(x)
    
    def forward(self, x):
        return self.Psi(x)

class eikonal_collective_variable(nn.Module): 
    def __init__(self, SDF, idx): 
        super(eikonal_collective_variable, self).__init__()
        self.SDF = SDF
        self.idx = idx
    def forward(self, x): 
        x.requires_grad_(True)
        output = self.SDF(x)
        grad = torch.autograd.grad([output.sum()], [x], create_graph=True)[0]
        return grad

## get modules 
def get_featurizer():
    selected_atoms = [3, 6, 9, 13]  # Example: select the carbons
    N = 14
    carbons_matrix = torch.zeros(len(selected_atoms), N)
    for i, atom in enumerate(selected_atoms):
        carbons_matrix[i, atom] = 1
    
    feature_map = RecenterBondLayer([6, 9], batch_mode=True)
    featurizer = featurizer_carbons(carbons_matrix, feature_map) 
    return featurizer

def get_chart(): 
    activation = nn.Tanh()
    input_dim = 12
    hidden1_dim = 32
    hidden2_dim = 32
    hidden3_dim = 32
    hidden4_dim = 32
    encoder_dim = 4
    output_dim = 42
    
    loader = True
    if loader: 
        model_encoder_reg = Encoder(activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim)
        model_encoder_reg.load_state_dict(torch.load(os.path.join(script_dir, "model_encoder_DNet_Laplacian_8oct2024_carbons")))
        model_encoder_reg.eval()
    for param in model_encoder_reg.parameters():
        param.requires_grad = False
    return model_encoder_reg

def get_SDF(): 
    activation_psi = periodic_activation()
    # activation_psi = nn.Tanh()
    input_dim_psi = 3
    hidden1_dim_psi = 30
    hidden2_dim_psi = 45
    hidden3_dim_psi = 32
    hidden4_dim_psi = 32
    potential_dim_psi = 1
    
    
    # regular models:
    # initializing the optimizer for the encoder
    loader = True
    
    if loader:
        model_Psi = PsiNetwork(activation_psi, input_dim_psi, hidden1_dim_psi, hidden2_dim_psi, hidden3_dim_psi, hidden4_dim_psi, potential_dim_psi)
        model_Psi.load_state_dict(torch.load(os.path.join(script_dir, 'potential_oct8_carbons_01')))
    for param in model_Psi.parameters(): 
        param_requires_grad = False
    return model_Psi 

def get_cv_1(): 
    model_Psi = get_SDF()
    eik_cv = eikonal_collective_variable(model_Psi, 0)
    return eik_cv

def get_cv_2(): 
    activation_psi = periodic_activation()
    # activation_psi = nn.Tanh()
    input_dim_psi = 3
    hidden1_dim_psi = 30
    hidden2_dim_psi = 45
    hidden3_dim_psi = 32
    hidden4_dim_psi = 32
    potential_dim_psi = 1
    loader = True
    if loader: 
        second_CV = PsiNetwork(periodic_activation(), input_dim_psi, hidden1_dim_psi, \
                           hidden2_dim_psi, hidden3_dim_psi, hidden4_dim_psi, potential_dim_psi)
        second_CV.load_state_dict(torch.load(os.path.join(script_dir, 'second_CV_cylinder_oct20')))
    for param in second_CV.parameters():
        param.requires_grad = False
    return second_CV

