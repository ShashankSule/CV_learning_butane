# Exploring the Butane Molecule

This project aims to explore and analyze the butane molecule through three primary components: **Sampling**, **Learning**, and **Estimation**. Each section outlines different methodologies and algorithms applied to understand the molecule’s structure and dynamics.

## 1. Sampling

### Overview
The Sampling component focuses on generating configurations of the butane molecule using Metadynamics simulations with OpenMM. We utilize biased potentials to explore a broader range of molecular conformations, overcoming high-energy barriers in the conformational space.

### Simulation Details
The primary sampling method is implemented through a Jupyter notebook: [`butane_openmm_metadynamics.ipynb`](https://github.com/ShashankSule/exploring_butane/blob/main/butane_openmm_metadynamics.ipynb). This notebook uses metadynamics to bias the simulation based on the dihedral angle θ.

### Free Energy Landscape
We visualize the free energy landscape as a function of the order parameter `(cos(θ), sin(θ))`, where θ is the dihedral angle of the butane molecule. This plot helps us understand the energetic preferences of the molecule's configurations:

![Free Energy Landscape](https://github.com/ShashankSule/exploring_butane/blob/main/figures/free_energy_landscape.png)

## 2. Learning

### Overview
This section involves leveraging machine learning techniques to analyze and predict key properties of butane configurations. We focus on learning low-dimensional embeddings of the sampled data and modeling it with neural networks.

### SO(3) Invariant Embedding
We employ **diffusion maps** on an SO(3)-invariant embedding of the butane data, achieved by aligning the C1-C2 bond to the x-axis. This embedding preserves the rotational symmetry of the molecule and facilitates efficient learning of intrinsic geometric structures:

![SO(3) Invariant Embedding](https://github.com/ShashankSule/exploring_butane/blob/main/figures/so3_invariant_embedding.png)

### Out-of-Sample Extension with Neural Networks
We further trained a neural network using **diffusion nets** to extend the learned embedding to out-of-sample configurations. This neural network efficiently predicts embeddings for new configurations based on their rotationally-invariant features:

![Diffusion Net](https://github.com/ShashankSule/exploring_butane/blob/main/figures/diffusion_net.png)

Curiously, the norm squared of the first two eigenvectors correlates heavily with the cosine of the dihedral angle:  

![Neural Network CV](https://github.com/ShashankSule/exploring_butane/blob/main/figures/nn_cv.png)

Using OpenMMTorch we compute the free energy landscape of this neural network-based CV. We had to use the binning method here: 

![Neural Network CV, Free energy landscape](https://github.com/ShashankSule/exploring_butane/blob/main/figures/nn_cv_free_energy.png)


## 3. Estimation

### Overview
The Estimation component utilizes statistical and numerical techniques to derive various properties of the butane molecule. We estimate free energies, diffusion tensors, committor functions, transition rates, and minimum energy paths.

### Properties Estimated
1. **Free Energies**: Calculated using the collective variable `(cos(θ), sin(θ))` to determine the relative stability of different configurations.

    ![Free Energy Plot](https://github.com/ShashankSule/exploring_butane/blob/main/figures/free_energy_plot.png)

2. **Diffusion Tensors**: Estimated from the sampled configurations to characterize the anisotropic diffusion behavior in the molecule's conformational space.

    ![Diffusion Tensor Plot](https://github.com/ShashankSule/exploring_butane/blob/main/figures/diffusion_tensor_plot.png)

3. **Committor Function**: Estimated to understand the probability of transitioning between key states.

    ![Committor Function Plot](https://github.com/ShashankSule/exploring_butane/blob/main/figures/committor_function_plot.png)

4. **Transition Rates**: Analyzed to identify the likelihood and speed of transitions between different conformations. We use the method utilizing diffusion maps outlined in Evans, Cameron and Tiwary 2021. The transition rate was found to be approximately 87/ps. 

5. **Minimum Energy Paths**: Calculated between significant configurations such as the **gauche-60**, **gauche-300** and **anti** configurations to identify the most probable transition paths. We compute the **gauche-60** --> **anti** and **anti** --> **gauche-300** paths, adjoin the two, and reparameterized the adjoined path to unit speed.

    ![Minimum Energy Path Plot](https://github.com/ShashankSule/exploring_butane/blob/main/figures/minimum_energy_path_plot.png)

## Getting Started

### Prerequisites
- **Python 3.8+** with libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `pytorch`.
- Molecular simulation tools like `OpenMM` or `MDTraj` for sampling.
  
### Installation
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/ShashankSule/exploring_butane.git
cd exploring_butane
pip install -r requirements.txt
```

### Running the Project
Each section is modular, with dedicated scripts for sampling, learning, and estimation. Each sampling has been illustrated as a tutorial in a jupyter notebook. 