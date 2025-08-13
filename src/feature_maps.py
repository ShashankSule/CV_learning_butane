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
from torch.autograd.gradcheck import gradcheck

class GramMatrixLayer(nn.Module):
    def __init__(self, mass_matrix):
        super(GramMatrixLayer, self).__init__()
        self.mass_matrix = mass_matrix

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1] % 3 == 0, "Input must have shape [batch_size, 3N]"
        N = x.shape[1] // 3
        x = x.reshape((batch_size, N, 3))   
        # Select the specified rows using the mass matrix
        x_selected = torch.matmul(self.mass_matrix, x)
        # Recenter the selected 3D vectors by subtracting their mean
        x_centered = x_selected - x_selected.mean(dim=1, keepdim=True)
        # Compute the Gram matrix for each batch element
        gram_matrices = torch.bmm(x_centered, x_centered.transpose(1, 2))
        return gram_matrices.reshape(batch_size, -1)
    
class RecenteringLayer(nn.Module):
    def __init__(self, mask_matrix):
        super(RecenteringLayer, self).__init__()
        self.mask_matrix = mask_matrix

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1] % 3 == 0, "Input must have shape [batch_size, 3N]"
        N = x.shape[1] // 3
        x = x.reshape((batch_size, N, 3))   
        # Select the specified rows using the mass matrix
        x_selected = torch.matmul(self.mask_matrix, x)
        # Recenter the selected 3D vectors by subtracting their mean
        x_centered = x_selected - x_selected.mean(dim=1, keepdim=True)
        return x_centered.reshape(batch_size, -1)
    
class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()  
    def forward(self, x):
        return x

class DihedralAngleLayer(nn.Module):
    def __init__(self, mask_matrix):
        super(DihedralAngleLayer, self).__init__()
        self.mask_matrix = mask_matrix

    def dihedral_angle_torch(self, x):
        # Select the specified atoms using the mask matrix
        carbons = torch.matmul(self.mask_matrix, x)
        vectors = carbons[1:,:] - carbons[:-1,:]
        na = torch.cross(-vectors[0,:], vectors[1,:])
        nb = torch.cross(-vectors[1,:], vectors[2,:])
        xx = torch.dot(na, nb)
        xp = torch.cross(na, nb)
        yy = torch.dot(vectors[1,:].T, xp) / torch.norm(vectors[1,:])
        angle = torch.atan2(yy, xx)
        return angle

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1] % 3 == 0, "Input must have shape [batch_size, 3N]"
        N = x.shape[1] // 3
        x = x.reshape((batch_size, N, 3))
        
        angles = torch.zeros(batch_size, device=x.device)
        for i in range(batch_size):
            angles[i] = self.dihedral_angle_torch(x[i])
        
        return angles


class RecenterBondLayer(nn.Module):
    def __init__(self, atom_ind, batch_mode=True):
        super(RecenterBondLayer, self).__init__()
        self.batch_mode = batch_mode
        self.ind = atom_ind 
    def recenter_bond_torch(self, x):
        assert x.shape[-1] % 3 == 0, "Input must have shape 3N"
        x = x.reshape((x.shape[-1] // 3, 3))        
        recentered_x = x - x[self.ind[0], :]
        theta = torch.atan2(recentered_x[self.ind[1], 1], recentered_x[self.ind[1], 0])
        
        xy_rotated = torch.tensor([
            [torch.cos(-theta), -torch.sin(-theta), 0],
            [torch.sin(-theta), torch.cos(-theta), 0],
            [0, 0, 1]
        ]) @ recentered_x.T
        
        xy_rotated = xy_rotated.T
        phi = torch.atan2(xy_rotated[self.ind[1], 2], xy_rotated[self.ind[1], 0])
        
        zx_rotated = torch.tensor([
            [torch.cos(-phi), 0, -torch.sin(-phi)],
            [0, 1, 0],
            [torch.sin(-phi), 0, torch.cos(-phi)]
        ]) @ xy_rotated.T
        
        zx_rotated = zx_rotated.T
        
        return zx_rotated.flatten()

    def recenter_bond_batch_torch(self, batch_x):
        batch_size = batch_x.shape[0]
        assert batch_x.shape[-1] % 3 == 0, "Input must have shape [batch_size, 3N]"
        batch_x = batch_x.reshape((batch_size, batch_x.shape[-1] // 3, 3)) # comment this out if you don't need it
        recentered_x = batch_x - batch_x[:, self.ind[0], :].unsqueeze(1)
        theta = torch.atan2(recentered_x[:, self.ind[1], 1], recentered_x[:, self.ind[1], 0])

        cos_theta = torch.cos(-theta)
        sin_theta = torch.sin(-theta)
        xy_rotation_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta, torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([sin_theta, cos_theta, torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([torch.zeros_like(cos_theta), torch.zeros_like(cos_theta), torch.ones_like(cos_theta)], dim=-1)
        ], dim=1)

        xy_rotated = torch.matmul(xy_rotation_matrix, recentered_x.transpose(1, 2)).transpose(1, 2)

        phi = torch.atan2(xy_rotated[:, self.ind[1], 2], xy_rotated[:, self.ind[1], 0])
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
            assert x.dim(), "Input must have shape [batch_size, 4, 3]"
            return self.recenter_bond_batch_torch(x)
        else:
            # If not batch mode, x should have shape [4, 3]
            # assert x.dim() == 2 and x.shape == (4, 3), "Input must have shape [4, 3]"
            assert x.dim() == 1, "Input must have shape [batch_size, 4, 3]"
            return self.recenter_bond_torch(x)
        
class OrthogonalChangeOfBasisBatchedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data):
        # Ensure that the input data is of shape (b, N)
        assert input_data.dim() == 2, "The input should be a 2-D tensor of shape (b, N)."
        # Reshape the input into points of shape (b, N//3, 3)
        b, N = input_data.shape
        points = input_data.view(b, -1, 3)
        points = points - points[:, 6, :].unsqueeze(1)
        # Extract the three points from indices 9, 6, and 13 for each batch element
        p1, p2, p3 = points[:, 9, :], points[:, 6, :], points[:, 13, :]
        # Calculate two vectors lying in the plane for each batch element
        v1 = p2 - p1
        v2 = p3 - p1
        # Calculate the normal vector to the plane using the cross product for each batch element
        normal_vector = torch.cross(v1, v2, dim=1)
        normal_vector = normal_vector / torch.norm(normal_vector, dim=1, keepdim=True)  # Normalize the normal vector
        # Normalize the vectors lying in the plane to form a basis for each batch element
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = v2 - torch.sum(v2 * v1, dim=1, keepdim=True) * v1  # Make v2 orthogonal to v1
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        # Form an orthonormal basis from v1, v2, and the normal vector for each batch element
        R = torch.stack([v1, v2, normal_vector], dim=1)  # Shape: (b, 3, 3)
        # Transform each row of the points by multiplying with the transpose of R for each batch element
        transformed_points = torch.bmm(points, R.transpose(1, 2))
        # Save R for potential debugging or backward calculations if needed
        ctx.save_for_backward(R)
        # Flatten each batch element and return as a 2-D tensor of shape (b, N)
        return transformed_points.view(b, -1)

class OrthogonalChangeOfBasisBatched(nn.Module):
    def forward(self, input_data):
        return OrthogonalChangeOfBasisBatchedFunction.apply(input_data)

class OrthogonalChangeOfBasisUnbatched(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_data):
        # Ensure that the input data is 1D
        assert input_data.dim() == 1, "The input should be a 1-D tensor."
        # Reshape the input into points of shape (N//3, 3)
        points = input_data.view(-1, 3)
        points = points - points[6]
        p1, p2, p3 = points[9], points[6], points[13]
        v1 = p2 - p1
        v2 = p3 - p1
        normal_vector = torch.cross(v1, v2)
        normal_vector = normal_vector / torch.norm(normal_vector)
        v1 = v1 / torch.norm(v1)
        v2 = v2 - torch.dot(v2, v1) * v1  # Make v2 orthogonal to v1
        v2 = v2 / torch.norm(v2)
        R = torch.stack([v1, v2, normal_vector])
        transformed_points = torch.matmul(points, R.t())
        return transformed_points.flatten()
        
class featurizer_carbons(nn.Module):
    def __init__(self, feature_map):
        super(featurizer_carbons, self).__init__()
        self.feature_map = feature_map
        self.carbon_indices = torch.tensor([3, 6, 9, 13])

    def forward(self, x):
        batch_size = x.shape[0]
        num_atoms = x.shape[1] // 3
        
        # Apply the initial feature map (e.g., alignment)
        aligned_config = self.feature_map(x).view(batch_size, num_atoms, 3)
        
        # Select carbon atoms
        carbons = torch.index_select(aligned_config, 1, self.carbon_indices.to(x.device))
        
        return carbons.flatten(start_dim=1)