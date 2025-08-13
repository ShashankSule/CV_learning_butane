import torch.nn as nn
import torch
import sys
import os
src_path = os.path.abspath('../../src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from feature_maps import OrthogonalChangeOfBasisUnbatched
from _dnet_architectures import standard_4_layer_dnet_tanh_encoder

class planealign_cv(nn.Module):
    def __init__(self, device): 
        super().__init__()
        # self.idx = idx
        self.featurizer = OrthogonalChangeOfBasisUnbatched()
        self.input_dim = 42
        self.encoder_dim = 2
        self.encoder = standard_4_layer_dnet_tanh_encoder(self.input_dim, self.encoder_dim).to(device)
        self.encoder = self.encoder.load_state_dict(torch.load("plane_align_diffusion_net.pth", map_location=device))
    def forward(self, input):
        features = self.featurizer(10.0*input.flatten()) # adjust nm to angstrom 
        CVs = self.encoder(features)
        return -torch.arctan2(CVs[...,1],CVs[...,0]) 

def get_planealign_cv(device):
    cv = planealign_cv(device)
    return cv



