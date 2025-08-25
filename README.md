# Learning Collective Variables That Preserve Transition Rates

This is the **official repository** for the paper [*Learning collective variables that preserve transition rates*](https://arxiv.org/abs/2506.01222), where we discover the dihedral angle of the butane molecule using invariant manifold learning and principles from quantitative coarse graining theory.

## Overview

This repository contains the complete implementation for discovering and validating collective variables (CVs) for molecular systems, specifically focusing on the butane molecule. Our approach combines invariant manifold learning with quantitative coarse graining theory to identify CVs that preserve essential dynamical properties, particularly transition rates between metastable states.

The repository is organized into three main modules: **Learn**, **Simulate**, and **Estimate**, each serving distinct but complementary roles in the collective variable discovery pipeline.

## Repository Structure

### 📚 **Learn Module** (`learn/`)

The Learn module is the **main module for learning the algorithm** and contains all machine learning components for discovering collective variables through various manifold learning techniques. 

#### **Data Source**

The primary butane trajectory data comes from [`learn/data/butane_nonaligned.npz`](learn/data/butane_nonaligned.npz). This file contains molecular dynamics trajectory data for the butane molecule:

```python
import numpy as np

# Load the butane trajectory data
data = np.load('learn/data/butane_nonaligned.npz')
print("Available keys:", list(data.keys()))
```

For generating new butane data at 300K, refer to the simulation notebook: [`simulate/butane_openmm_300K.ipynb`](simulate/butane_openmm_300K.ipynb).

#### **Three-Step Learning Algorithm**

The main learning algorithm is broken down into three sequential steps:

#### **Step 1: Manifold Learning** (`learn/manifold_learning/`)

a. **Feature Map Selection**: Use [`butane_ies.py`](learn/manifold_learning/butane_ies.py) to pick optimal dimensions or feature map representations through independent eigencoordinate selection (IES). Core IES algorithms are in [`src/ies/`](src/ies/) (forked from [ic-pml repository](https://github.com/he-jesse/ic-pml)).

b. **Diffusion Map Construction**: Once the feature map is selected, generate a diffusion map using [`construct_dmap.py`](learn/manifold_learning/construct_dmap.py) with configuration from [`config_dmap.yml`](learn/manifold_learning/config_dmap.yml). Results are stored in [`dmaps/`](learn/manifold_learning/dmaps/) directory.

c. **Diffusion Net Training**: After computing the diffusion map, train a diffusion net using [`learn_DNET.py`](learn/manifold_learning/learn_DNET.py) for out-of-sample extension. Trained models are stored in [`dnets/`](learn/manifold_learning/dnets/) directory with timestamps and configurations.

Additional components:
- [`feature_maps.py`](src/feature_maps.py) - Implementation of various feature map representations (BondAlign, GramMatrix, etc.)
- [`training_dnet.py`](learn/manifold_learning/training_dnet.py) - Core training utilities for diffusion nets
- [`compute_normals_with_dnet.py`](learn/manifold_learning/compute_normals_with_dnet.py) - Script for computing normal vectors using trained diffusion nets

#### **Step 2: Surrogate Potential Learning** (`learn/surrogate_potential/`)

a. **Data Loading**: Load the trajectory data and trained diffusion net using the specific dataloader class from [`src/_surrogate_potential_datasets.py`](src/_surrogate_potential_datasets.py).

b. **Training**: Train the surrogate potential using a combination of eikonal and local pointcloud normal losses through [`train_surrogate_potential.py`](learn/surrogate_potential/train_surrogate_potential.py).

c. **Example Implementation**: A complete example has been provided in [`surrogate_potential_bondalign23.ipynb`](learn/surrogate_potential/surrogate_potential_bondalign23.ipynb).

Additional components:
- [`config.yml`](learn/surrogate_potential/config.yml) - Configuration for surrogate potential training
- [`outputs/`](learn/surrogate_potential/outputs/) - Trained surrogate potential models

#### **Step 3: Collective Variable Learning** (`learn/collective_variable/`)

a. **Dataset Construction**: Once the surrogate potential has been trained, construct a dataset using the feature map, diffusion net, and the surrogate potential.

b. **CV Training**: Train the collective variable using the trainer in [`train_cv.py`](learn/collective_variable/train_cv.py) with configuration from [`config.yaml`](learn/collective_variable/config.yaml).

c. **Implementation Examples**: A complete example has been provided in [`second_CV_bondalign.ipynb`](learn/collective_variable/second_CV_bondalign.ipynb). Moreover, trained CVs and loading instructions are available in the simulation notebooks: [`simulate/feature_map_align_12_carbons/process_CV.ipynb`](simulate/feature_map_align_12_carbons/process_CV.ipynb) and [`simulate/feature_map_plane_align/process_CVs.ipynb`](simulate/feature_map_plane_align/process_CVs.ipynb).

Additional components:
- [`checkpoints/`](learn/collective_variable/checkpoints/) - Saved model checkpoints
- [`logs/`](learn/collective_variable/logs/) - Training logs and curves

#### **Additional Experiments**

**LAPCAE Experiments** (`learn/manifold_learning/`):
- [`LAPCAE.ipynb`](learn/manifold_learning/LAPCAE.ipynb) - Laplacian Autoencoder experiments
- [`outputs/bondalign_23/`](learn/manifold_learning/outputs/bondalign_23/) - LAPCAE results for bond alignment feature map (2,3)
- **Generates Figure 10** from the paper (LAPCAE visualization)


### 🎯 **Simulate Module** (`simulate/`)

The Simulate module handles molecular dynamics simulations and free energy calculations.

#### **Simulation Setup**

- [`data/`](simulate/data/) - Contains butane molecular structure files (PDB, PSF, parameters)
- Configuration files for different CV representations:
  - [`config_bondalign_23.yaml`](simulate/config_bondalign_23.yaml) - Bond alignment (2,3) configuration
  - [`config_cos_dihedral.yaml`](simulate/config_cos_dihedral.yaml) - Cosine dihedral angle configuration  
  - [`config_dihedral_angle.yaml`](simulate/config_dihedral_angle.yaml) - Direct dihedral angle configuration
  - [`config_planealign.yaml`](simulate/config_planealign.yaml) - Plane alignment configuration
  - [`config_sin_cos_dihedral.yaml`](simulate/config_sin_cos_dihedral.yaml) - Sine-cosine dihedral configuration

#### **Free Energy and Dynamics**

- [`free_energy.py`](simulate/free_energy.py) - Script for computing free energy surfaces
- [`diffusion_tensors.py`](simulate/diffusion_tensors.py) - Computation of diffusion tensors in CV space
- [`output/`](simulate/output/) - Simulation results and analysis outputs

#### **Feature Map Processing**

- [`feature_map_align_12_carbons/`](simulate/feature_map_align_12_carbons/) - Processing scripts for bondalign(2,3) feature map
- [`feature_map_plane_align/`](simulate/feature_map_plane_align/) - Processing scripts for PlaneAlign feature map
- Notebooks for visualizing learned collective variables:
  - [`process_CV.ipynb`](simulate/feature_map_align_12_carbons/process_CV.ipynb) - Main CV processing and visualization
  - [`process_CVs.ipynb`](simulate/feature_map_plane_align/process_CVs.ipynb) - Additional CV analysis

### 📊 **Estimate Module** (`estimate/`)

The Estimate module focuses on computing transition rates and validating collective variables.

#### **Transition Analysis**

- [`transition_analysis_2D.ipynb`](estimate/transition_analysis_2D.ipynb) - **Generates Figure 3** and **Tables 1 & 2** from the paper
- [`transition_rates.ipynb`](estimate/transition_rates.ipynb) - Detailed transition rate calculations and visualizations

#### **Data**

- [`data/`](estimate/data/) - Contains preprocessed data files:
  - [`cosdihedral_data.npz`](estimate/data/cosdihedral_data.npz) - Cosine dihedral angle data
  - [`cossin.npz`](estimate/data/cossin.npz) - Combined cosine-sine data
  - [`dihedral_data.npz`](estimate/data/dihedral_data.npz) - Raw dihedral angle data
  - [`PlaneAlignTanh.npz`](estimate/data/PlaneAlignTanh.npz) - Plane alignment with tanh transformation data


## Figure Generation Guide

This repository can reproduce all main figures from the paper:

- **Figures 7 & 8**: Can be directly generated from [`additional_figs/embeddings/plotter.ipynb`](additional_figs/embeddings/plotter.ipynb) and [`additional_figs/lapcae_fig_7/visualize.m`](additional_figs/lapcae_fig_7/visualize.m)
- **Figure 9**: Run IES scripts in [`src/ies/`](src/ies/) and [`learn/manifold_learning/butane_ies.py`](learn/manifold_learning/butane_ies.py)
- **Figure 10**: Execute LAPCAE experiments in [`learn/manifold_learning/LAPCAE.ipynb`](learn/manifold_learning/LAPCAE.ipynb) with bondalign_23 configuration
- **Figure 3 & Tables 1, 2**: Use transition analysis notebooks in [`estimate/transition_analysis_2D.ipynb`](estimate/transition_analysis_2D.ipynb) and [`estimate/transition_rates.ipynb`](estimate/transition_rates.ipynb)

## Installation and Setup

### Prerequisites

- **Python 3.11.5** 
- **PyTorch** (for neural network components)
- **OpenMM** (for molecular dynamics)
- **NumPy, SciPy, Matplotlib** (standard scientific libraries)
- **MDTraj** (for trajectory analysis)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ShashankSule/CV-learning-butane.git
cd CV_learning_butane
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate butane_env
```


Note: Environment files are located at [`environment.yml`](environment.yml) and [`openmm_env.yml`](openmm_env.yml).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{sule2025learning,
  title={Learning collective variables that preserve transition rates},
  author={Sule, Shashank and Mehta, Arnav and Cameron, Maria K},
  journal={arXiv preprint arXiv:2506.01222},
  year={2025}
}
```
