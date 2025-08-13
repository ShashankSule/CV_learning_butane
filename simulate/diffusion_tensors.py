
import os, math, sys
from sys import stdout

import numpy as np
import matplotlib.pyplot as plt
import openmm.app  as omm_app
import openmm as omm
import simtk.unit as unit
from tqdm import tqdm

import torch.nn as nn
import torch
# import openmmtools
from openmmtorch import TorchForce
from copy import deepcopy
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import scipy.ndimage

from MDAnalysis.analysis.dihedrals import Dihedral

# Indicates whether to run a simulation or not
SIMULATE = False
device = "cuda:0"

# openmm system of butane

psf = omm_app.CharmmPsfFile('../data/butane.psf')
pdb = omm_app.PDBFile('../data/butane.pdb')
topology = psf.topology
params = omm_app.CharmmParameterSet('../data/top_all35_ethers.rtf',
                                    '../data/par_all35_ethers.prm')
system = psf.createSystem(params, nonbondedMethod=omm_app.NoCutoff)
with open("../output/system.xml", 'w') as file_handle:
    file_handle.write(omm.XmlSerializer.serialize(system))
## read the OpenMM system of butane
with open("../output/system.xml", 'r') as file_handle:
    xml = file_handle.read()
system = omm.XmlSerializer.deserialize(xml)
## read psf and pdb file of butane
psf = omm_app.CharmmPsfFile("../data/butane.psf")
pdb = omm_app.PDBFile('../data/butane.pdb')
topology = psf.topology

print(f'Finished butane setup!')
# dnet based cv
from models_features import OrthogonalChangeOfBasisUnbatched, Encoder_deep, periodic_activation # this defines the structure of the model
activation = nn.Tanh()
input_dim = 42
hidden1_dim = 32
hidden2_dim = 32
hidden3_dim = 32
hidden4_dim = 32
encoder_dim = 2
output_dim = 42
loader = True
if loader: 
    model_encoder_reg = Encoder_deep(activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim).to(device)
    model_encoder_reg.load_state_dict(torch.load("../../butane_AE/dnets/model_20250507_200144/model_encoder_state_dict.pth", map_location=device))
    model_encoder_reg.eval()
class CV_test(nn.Module):
    def __init__(self, idx, featurizer, CV): 
        super().__init__()
        self.idx = idx
        self.featurizer = featurizer
        self.CV = CV
    def forward(self, input):
        features = self.featurizer(10.0*input.flatten().to(torch.float32))
        CVs = self.CV(features.to(torch.float32))
        return torch.arctan2(CVs[...,1], CVs[...,0])
layer = OrthogonalChangeOfBasisUnbatched()
cv_model_1 = CV_test(0, layer, model_encoder_reg).to(device)
# cv_model_2 = CV_test(1, layer, model_encoder_reg).to(device)
cv1 = torch.jit.script(cv_model_1)
# cv2 = torch.jit.script(cv_model_2)
CV_force_1 = TorchForce(cv1)
# CV_force_2 = TorchForce(cv2)


print(f'Loaded up models, added to cv!')

## add a harmonic biasing potential on butane dihedral to the OpenMM system
cv_1 = omm.CustomCVForce("cv1")
cv_1.addCollectiveVariable("cv1",CV_force_1)

# cv_2 = omm.CustomCVForce("cv2")
# cv_2.addCollectiveVariable("cv2",CV_force_2)

## add a harmonic biasing potential on butane dihedral to the OpenMM system
bias_cv = omm.CustomCVForce('0.5*Kappa*diff^2; diff=abs(theta1 - x1)')
bias_cv.addGlobalParameter("Kappa", 500.0)
bias_cv.addGlobalParameter("x1", 0.0)
# bias_cv.addGlobalParameter("x2", 0.0)
bias_cv.addCollectiveVariable('theta1',cv_1)
# bias_cv.addCollectiveVariable("theta2",cv_2)
system.addForce(bias_cv)


# # 2. Set up and get data

#### setup an OpenMM context

## platform
platform = omm.Platform.getPlatformByName('CUDA')

## integrator
# T = 298.15 * unit.kelvin  ## temperature
T = 300 * unit.kelvin
kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.AVOGADRO_CONSTANT_NA 
kbT = kbT.value_in_unit(unit.kilojoule_per_mole)

kbT_roomtemp = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*300.0*unit.kelvin
kbT_roomtemp = kbT_roomtemp.value_in_unit(unit.kilojoule_per_mole)


## integrator
fricCoef = 10/unit.picoseconds ## friction coefficient 
# stepsize = 2 * unit.femtoseconds ## integration step size
stepsize = 1 * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

## construct an OpenMM context
# context = omm.Context(system, integrator, platform)
simulation = omm_app.Simulation(topology, system, integrator, platform)

print(f'Set up simulation!')
# Next we get data via `mdtraj`

mobile = mda.Universe("../data/butane.psf", "../output/traj_current/traj_constant_bias.dcd")
trajectory = torch.Tensor(mobile.trajectory.timeseries(mobile.atoms))
data = torch.zeros((len(mobile.trajectory), 1))
for i in tqdm(range(len(mobile.trajectory))):
    mobile.trajectory[i]
    x = 0.1*torch.tensor(mobile.atoms.positions, dtype=torch.float64).to(device) # multiply the input by 0.1 because the dcd writer saves everything by multiplying by 10. 
    data[i,0] = cv_model_1(x)
    # data[i,1] = cv_model_2(x)
N = data.shape[0]
N_samples = 10000
period = N // N_samples
# subsample the trajectory
evaluation_pts = np.linspace(-np.pi,np.pi,1000)
data_CV = data.detach().numpy()
# subsampled_traj = mobile.trajectory


# # 3a. Sampling short trajectories: 1D CV

simulation.integrator.setStepSize(1e-4*unit.femtoseconds)
simulation.context.setParameter("Kappa",1000.0)

print(f'Now starting trajectory bursts...')
sample_bursts = True

if sample_bursts:
    ## the main loop to run umbrella sampling window by window
    for index in tqdm(range(evaluation_pts.shape[0])):

        ## set the center of the biasing potential
        simulation.context.setParameter("x1", evaluation_pts[index])
    
        ## set current simulation state
        current_point = evaluation_pts[index]
        toroidal_distance = np.abs(current_point - data_CV)
        config_index = np.argmin(toroidal_distance)
        mobile.trajectory[config_index]
        print(data_CV[config_index])
        simulation.context.setPositions(0.1*mobile.atoms.positions)
        
        ## sampling production. trajectories are saved in dcd files
        file_handle = open(f"../output/traj_current/short_traj_planealign/traj_{index}.dcd", 'bw')
        dcd_file = omm_app.dcdfile.DCDFile(file_handle, psf.topology, dt = stepsize)
        for k in range(300):
            integrator.step(5)
            # print("saving...")
            state = simulation.context.getState(getPositions = True)
            positions = state.getPositions()
            dcd_file.writeModel(positions)
        file_handle.close()
        # if index > 10:
        #    break
print(f'Computing trajectories completed. Now computing diffusion tensor...')

def compute_grads(x):
    x.requires_grad_(True)
    features = layer(x)
    j_layer = []
    for i in range(features.shape[0]):
        jj = torch.autograd.grad(features[i].sum(), x, retain_graph=True)[0]
        j_layer.append(jj.unsqueeze(1))
    j_layer = torch.cat(tuple(j_layer),dim=1)
    y = model_encoder_reg(features)
    z = torch.arctan2(y[1],y[0])
    j_model = torch.autograd.grad(z,features,retain_graph=True)[0]
    total_grad = j_model.unsqueeze(0).detach()@j_layer.detach().T
    return total_grad
    
# # 4a. Analyze traj for 1D CV
sqrt_masses = [np.sqrt(system.getParticleMass(i)._value) for i in range(system.getNumParticles())]
sqrt_masses_tensor = ((torch.Tensor(sqrt_masses).unsqueeze(1))@torch.ones((1,3))).flatten().unsqueeze(0)
M = []
# i = 1
for i in tqdm(range(evaluation_pts.shape[0])):

   # get short traj data 
    ref = mda.Universe("../data/butane.psf", f"../output/traj_current/short_traj_planealign/traj_{i}.dcd", in_memory=True)
    coordinates = torch.tensor(ref.trajectory.timeseries(ref.atoms)).to(device)
    coords = coordinates.permute(1,0,2).flatten(start_dim=1)

    # set up jacobians 
    j_norm = 0.0
    for j in range(coords.shape[0]):
         j_norm += torch.norm(compute_grads(coords[j,:])*(1/sqrt_masses_tensor.to(device)))**2
    diff_tensor = (1/coords.shape[0])*j_norm
    M.append(diff_tensor.cpu().numpy())

diffusion_tensors = np.array(M)
diffusion_tensors = scipy.ndimage.gaussian_filter1d(diffusion_tensors,\
                                                  16.0, mode='wrap')
fname = f"../output/traj_current/diffusion_tensors.npz"
np.savez(fname, evaluation_points=evaluation_pts, diffusion_tensors=diffusion_tensors)

