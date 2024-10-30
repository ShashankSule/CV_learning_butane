import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import copy
import pandas as pd
from tqdm import tqdm
import numpy as np 
import scipy.sparse as sps
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
import datetime
import argparse
from IPython.display import clear_output

def recenter_bond(x):
    x = x.reshape((4,3))
    assert x.shape == (4,3)
    recentered_x = x - x[1,:]
    theta = np.arctan2(recentered_x[2,1], recentered_x[1,0])
    xy_rotated = np.array([[np.cos(-theta), -np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])@recentered_x.T
    xy_rotated = xy_rotated.T
    phi = np.arctan2(xy_rotated[2,2], xy_rotated[2,0])
    zx_rotated = np.array([[np.cos(-phi),0, -np.sin(-phi)],[0, 1, 0], [np.sin(-phi), 0, np.cos(-phi)]])@xy_rotated.T
    zx_rotated = zx_rotated.T
    return zx_rotated.flatten()

def recenter_bond_torch(x):
    x = x.reshape((4,3))
    assert x.shape == (4,3)
    recentered_x = x - x[1,:]
    theta = np.arctan2(recentered_x[2,1], recentered_x[2,0])
    xy_rotated = np.array([[np.cos(-theta), -np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])@recentered_x.T
    xy_rotated = xy_rotated.T
    phi = np.arctan2(xy_rotated[2,2], xy_rotated[2,0])
    zx_rotated = np.array([[np.cos(-phi),0, -np.sin(-phi)],[0, 1, 0], [np.sin(-phi), 0, np.cos(-phi)]])@xy_rotated.T
    zx_rotated = zx_rotated.T
    return zx_rotated.flatten()

def dihedral_angle_torch(x):
    # assert x.shape == torch.Size([14,3])
    # carbons = x[mask,:]
    carbons = x
    vectors = carbons[1:,:] - carbons[:-1,:]
    na = torch.cross(-vectors[0,:], vectors[1,:])
    nb = torch.cross(-vectors[1,:], vectors[2,:])
    xx = torch.dot(na, nb)
    xp = torch.cross(na,nb)
    yy = torch.dot(vectors[1,:].T,xp)/torch.norm(vectors[1,:])
    angle = torch.atan2(yy,xx)
    return angle


class RecenterBondLayer(nn.Module):
    def __init__(self, atom_ind, batch_mode=True):
        super(RecenterBondLayer, self).__init__()
        self.batch_mode = batch_mode
        self.ind = atom_ind 
    def recenter_bond_torch(self, x):
        x = x.reshape((4, 3))
        assert x.shape == (4, 3)
        
        recentered_x = x - x[self.ind, :]
        theta = torch.atan2(recentered_x[self.ind+1, 1], recentered_x[self.ind+1, 0])
        
        xy_rotated = torch.tensor([
            [torch.cos(-theta), -torch.sin(-theta), 0],
            [torch.sin(-theta), torch.cos(-theta), 0],
            [0, 0, 1]
        ]) @ recentered_x.T
        
        xy_rotated = xy_rotated.T
        phi = torch.atan2(xy_rotated[self.ind+1, 2], xy_rotated[self.ind+1, 0])
        
        zx_rotated = torch.tensor([
            [torch.cos(-phi), 0, -torch.sin(-phi)],
            [0, 1, 0],
            [torch.sin(-phi), 0, torch.cos(-phi)]
        ]) @ xy_rotated.T
        
        zx_rotated = zx_rotated.T
        
        return zx_rotated.flatten()

    def recenter_bond_batch_torch(self, batch_x):
        batch_size = batch_x.shape[0]
        batch_x = batch_x.reshape((batch_size, 4, 3)) # comment this out if you don't need it
        recentered_x = batch_x - batch_x[:, self.ind, :].unsqueeze(1)
        theta = torch.atan2(recentered_x[:, self.ind+1, 1], recentered_x[:, self.ind+1, 0])

        cos_theta = torch.cos(-theta)
        sin_theta = torch.sin(-theta)
        xy_rotation_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta, torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([sin_theta, cos_theta, torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([torch.zeros_like(cos_theta), torch.zeros_like(cos_theta), torch.ones_like(cos_theta)], dim=-1)
        ], dim=1)

        xy_rotated = torch.matmul(xy_rotation_matrix, recentered_x.transpose(1, 2)).transpose(1, 2)

        phi = torch.atan2(xy_rotated[:, self.ind+1, 2], xy_rotated[:, self.ind+1, 0])
        cos_phi = torch.cos(-phi)
        sin_phi = torch.sin(-phi)
        zx_rotation_matrix = torch.stack([
            torch.stack([cos_phi, torch.zeros_like(cos_phi), -sin_phi], dim=-1),
            torch.stack([torch.zeros_like(cos_phi), torch.ones_like(cos_phi), torch.zeros_like(cos_phi)], dim=-1),
            torch.stack([sin_phi, torch.zeros_like(cos_phi), cos_phi], dim=-1)
        ], dim=1)

        zx_rotated = torch.matmul(zx_rotation_matrix, xy_rotated.transpose(1, 2)).transpose(1, 2)
        return zx_rotated.reshape(batch_size, -1)

    def forward(self, x):
        if self.batch_mode:
            # If batch mode, x should have shape [batch_size, 4, 3]
            # assert x.dim() == 3 and x.shape[1:] == (4, 3), "Input must have shape [batch_size, 4, 3]"
            assert x.dim() == 2 and x.shape[1] == 12, "Input must have shape [batch_size, 4, 3]"
            return self.recenter_bond_batch_torch(x)
        else:
            # If not batch mode, x should have shape [4, 3]
            # assert x.dim() == 2 and x.shape == (4, 3), "Input must have shape [4, 3]"
            assert x.dim() == 1 and x.shape[0] == 12, "Input must have shape [batch_size, 4, 3]"
            return self.recenter_bond_torch(x)