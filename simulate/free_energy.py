#!/usr/bin/env python
"""
Free energy calculation script using metadynamics with various collective variables.
Supports multiple CV types including:
- Dihedral angles (using OpenMM CustomTorsionForce)
- Bond-aligned CVs (using get_cv_1 and get_cv_2 from cv_loader.py)
- Plane-aligned CVs (using get_planealign_cv from cv_loader.py)
Configuration is loaded from free_energy_config.yaml.
"""

import os
import sys
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt

import openmm.app as omm_app
import openmm as omm
import simtk.unit as unit
from tqdm import tqdm

import torch
import torch.nn as nn
from openmmtorch import TorchForce
from copy import deepcopy
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.dihedrals import Dihedral

# Add current directory to path for CV loaders
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    import feature_map_align_12_carbons.cv_loader as carbons_cv_loader
except ImportError:
    print("Warning: Could not import carbons CV loader")
    carbons_cv_loader = None

try:
    import feature_map_plane_align.cv_loader as plane_cv_loader
except ImportError:
    print("Warning: Could not import plane CV loader")
    plane_cv_loader = None

# Simplified CV implementations using OpenMM native forces


# Neural network CVs are loaded directly from cv_loader modules


def load_config(config_path="free_energy_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_butane_system(config):
    """Set up the butane OpenMM system."""
    system_config = config['system']
    
    # Load topology and coordinates
    psf = omm_app.CharmmPsfFile(system_config['psf_file'])
    pdb = omm_app.PDBFile(system_config['pdb_file'])
    topology = psf.topology
    
    # Load force field parameters
    params = omm_app.CharmmParameterSet(
        system_config['ff_rtf_file'],
        system_config['ff_prm_file']
    )
    
    # Create system
    system = psf.createSystem(params, nonbondedMethod=omm_app.NoCutoff)
    
    return system, topology, pdb


def create_collective_variables(config):
    """Create collective variables based on configuration."""
    cv_type = config['cv_type']
    cv_params = config['cv_params']
    device = config['device']
    
    collective_variables = []
    bias_variables = []
    
    if cv_type == "dihedral":
        # Use CustomTorsionForce for dihedral angle
        cv = omm.CustomTorsionForce("theta")
        dihedral_atoms = cv_params['dihedral_atoms']
        cv.addTorsion(dihedral_atoms[0], dihedral_atoms[1], dihedral_atoms[2], dihedral_atoms[3])
        
        omm_cv = omm.CustomCVForce("dihedral")
        omm_cv.addCollectiveVariable("dihedral", cv)
        collective_variables.append(omm_cv)
        
        # Create bias variable
        cv_range = cv_params['single_cv_range']
        bias_var = omm_app.metadynamics.BiasVariable(
            force=omm_cv,
            minValue=cv_range[0],
            maxValue=cv_range[1],
            biasWidth=config['metadynamics']['bias_width'],
            periodic=True,  # Dihedral angles are periodic
            gridWidth=config['metadynamics']['grid_width']
        )
        bias_variables.append(bias_var)
        
    elif cv_type == "cos_dihedral":
        # Use CustomTorsionForce for cosine of dihedral angle
        cv = omm.CustomTorsionForce("cos(theta)")
        dihedral_atoms = cv_params['dihedral_atoms']
        cv.addTorsion(dihedral_atoms[0], dihedral_atoms[1], dihedral_atoms[2], dihedral_atoms[3])
        
        omm_cv = omm.CustomCVForce("cos_dihedral")
        omm_cv.addCollectiveVariable("cos_dihedral", cv)
        collective_variables.append(omm_cv)
        
        bias_var = omm_app.metadynamics.BiasVariable(
            force=omm_cv,
            minValue=-1.0,
            maxValue=1.0,
            biasWidth=config['metadynamics']['bias_width'],
            periodic=False,
            gridWidth=config['metadynamics']['grid_width']
        )
        bias_variables.append(bias_var)
        
    elif cv_type == "sin_cos_dihedral":
        # Create sin and cos CVs using CustomTorsionForce
        dihedral_atoms = cv_params['dihedral_atoms']
        
        cv_sin = omm.CustomTorsionForce("sin(theta)")
        cv_cos = omm.CustomTorsionForce("cos(theta)")
        cv_sin.addTorsion(dihedral_atoms[0], dihedral_atoms[1], dihedral_atoms[2], dihedral_atoms[3])
        cv_cos.addTorsion(dihedral_atoms[0], dihedral_atoms[1], dihedral_atoms[2], dihedral_atoms[3])
        
        omm_sin_cv = omm.CustomCVForce("sin_dihedral")
        omm_sin_cv.addCollectiveVariable("sin_dihedral", cv_sin)
        
        omm_cos_cv = omm.CustomCVForce("cos_dihedral")
        omm_cos_cv.addCollectiveVariable("cos_dihedral", cv_cos)
        
        collective_variables.extend([omm_sin_cv, omm_cos_cv])
        
        # Create bias variables
        for omm_cv, name in [(omm_sin_cv, "sin"), (omm_cos_cv, "cos")]:
            bias_var = omm_app.metadynamics.BiasVariable(
                force=omm_cv,
                minValue=-1.0,
                maxValue=1.0,
                biasWidth=config['metadynamics']['bias_width'],
                periodic=False,
                gridWidth=config['metadynamics']['grid_width']
            )
            bias_variables.append(bias_var)
            
    elif cv_type == "bondalign":
        # Use get_cv_1 and get_cv_2 from carbons CV loader directly
        if carbons_cv_loader is None:
            raise ImportError("Carbons CV loader not available")
            
        cvs_to_use = []
        if cv_params.get('use_cv_1', False):
            cv_1 = carbons_cv_loader.get_cv_1()
            cvs_to_use.append((cv_1, 'cv_1'))
            
        if cv_params.get('use_cv_2', False):
            cv_2 = carbons_cv_loader.get_cv_2()
            cvs_to_use.append((cv_2, 'cv_2'))
            
        for cv, name in cvs_to_use:
            # Convert pre-existing CV directly to JIT and TorchForce
            cv.to(device)
            cv_torch = torch.jit.script(cv)
            cv_force = TorchForce(cv_torch)
            
            omm_cv = omm.CustomCVForce(name)
            omm_cv.addCollectiveVariable(name, cv_force)
            collective_variables.append(omm_cv)
            
            # Create bias variable
            if name == 'cv_1':
                cv_range = cv_params['cv_1_range']
            else:
                cv_range = cv_params['cv_2_range']
                
            bias_var = omm_app.metadynamics.BiasVariable(
                force=omm_cv,
                minValue=cv_range[0],
                maxValue=cv_range[1],
                biasWidth=config['metadynamics']['bias_width'],
                periodic=False,
                gridWidth=config['metadynamics']['grid_width']
            )
            bias_variables.append(bias_var)
            
    elif cv_type == "planealign":
        # Use get_planealign_cv from plane CV loader directly
        if plane_cv_loader is None:
            raise ImportError("Plane CV loader not available")
            
        cv = plane_cv_loader.get_planealign_cv(device)
        cv.to(device)
        
        cv_torch = torch.jit.script(cv)
        cv_force = TorchForce(cv_torch)
        
        omm_cv = omm.CustomCVForce("planealign")
        omm_cv.addCollectiveVariable("planealign", cv_force)
        collective_variables.append(omm_cv)
        
        cv_range = cv_params['single_cv_range']
        bias_var = omm_app.metadynamics.BiasVariable(
            force=omm_cv,
            minValue=cv_range[0],
            maxValue=cv_range[1],
            biasWidth=config['metadynamics']['bias_width'],
            periodic=False,
            gridWidth=config['metadynamics']['grid_width']
        )
        bias_variables.append(bias_var)
        
    else:
        raise ValueError(f"Unknown CV type: {cv_type}")
    
    return collective_variables, bias_variables


def run_metadynamics_simulation(config):
    """Run the complete metadynamics simulation."""
    print("Setting up butane system...")
    system, topology, pdb = setup_butane_system(config)
    
    print("Creating collective variables...")
    collective_variables, bias_variables = create_collective_variables(config)
    
    # Add CV forces to system
    for cv in collective_variables:
        system.addForce(cv)
    
    # Set up metadynamics
    sim_config = config['simulation']
    meta_config = config['metadynamics']
    
    T = sim_config['temperature'] * unit.kelvin
    meta = omm_app.Metadynamics(
        system,
        variables=bias_variables,
        temperature=T,
        biasFactor=meta_config['bias_factor'],
        height=meta_config['height'],
        frequency=meta_config['frequency'],
        saveFrequency=meta_config['frequency'],
        biasDir=config['output']['base_dir']
    )
    
    # Set up integrator and simulation
    platform = omm.Platform.getPlatformByName(sim_config['platform'])
    fricCoef = sim_config['friction_coeff'] / unit.picoseconds
    stepsize = sim_config['timestep'] * unit.femtoseconds
    integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)
    simulation = omm_app.Simulation(topology, system, integrator, platform)
    
    # Set initial positions
    simulation.context.setPositions(pdb.positions)
    if hasattr(pdb, 'boxVectors') and pdb.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*pdb.boxVectors)
    
    # Set up reporters
    output_config = config['output']
    base_dir = output_config['base_dir']
    os.makedirs(base_dir, exist_ok=True)
    
    dcd_file = os.path.join(base_dir, output_config['trajectory_file'])
    state_file = os.path.join(base_dir, output_config['state_file'])
    
    simulation.reporters.append(
        omm_app.DCDReporter(dcd_file, output_config['report_frequency'], False)
    )
    simulation.reporters.append(
        omm_app.StateDataReporter(
            state_file, output_config['report_frequency'],
            step=True, potentialEnergy=True
        )
    )
    simulation.reporters.append(
        omm_app.StateDataReporter(
            sys.stdout, output_config['console_report_frequency'],
            step=True, potentialEnergy=True
        )
    )
    
    print("Starting metadynamics simulation...")
    print(f"Running for {sim_config['total_steps']} steps")
    print(f"Temperature: {sim_config['temperature']} K")
    print(f"CV type: {config['cv_type']}")
    
    # Run simulation
    meta.step(simulation, sim_config['total_steps'])
    
    # Save results
    free_energy_file = os.path.join(base_dir, output_config['free_energy_file'])
    cvs_file = os.path.join(base_dir, output_config['cvs_file'])
    
    np.savez(free_energy_file, free_energy=meta.getFreeEnergy())
    np.savez(cvs_file, CVs=meta.getCollectiveVariables(simulation))
    
    # Save system parameters
    system_params = {
        'cv_type': config['cv_type'],
        'biasFactor': meta_config['bias_factor'],
        'biasWidth': meta_config['bias_width'],
        'T': T.value_in_unit(T.unit),
        'height': meta_config['height'],
        'frequency': meta_config['frequency'],
        'fricCoef': {
            'value': fricCoef.value_in_unit(fricCoef.unit),
            'unit': fricCoef.unit.get_symbol()
        },
        'stepsize': {
            'value': stepsize.value_in_unit(stepsize.unit),
            'unit': stepsize.unit.get_symbol()
        }
    }
    
    params_file = os.path.join(base_dir, output_config['params_file'])
    with open(params_file, 'w') as outfile:
        yaml.dump(system_params, outfile, default_flow_style=False, sort_keys=False)
    
    print(f"\nSimulation completed!")
    print(f"Results saved to: {base_dir}")
    print(f"Free energy: {free_energy_file}")
    print(f"CVs: {cvs_file}")
    print(f"Parameters: {params_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run metadynamics free energy calculation')
    parser.add_argument('--config', '-c', default='free_energy_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--simulate', action='store_true', default=True,
                       help='Whether to run simulation (default: True)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found!")
        return
    
    config = load_config(args.config)
    
    if args.simulate:
        run_metadynamics_simulation(config)
    else:
        print("Simulation skipped (--simulate not specified)")


if __name__ == "__main__":
    main()
