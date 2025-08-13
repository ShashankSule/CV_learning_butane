# Manifold Normal Computation Script

This script takes molecular dynamics data, maps it forward through a trained DNet model, and computes pointwise normals on the resulting manifold surface using Open3D.

## Overview

The script performs the following steps:

1. **Load Data**: Loads butane molecular dynamics trajectory data from an NPZ file
2. **Feature Mapping**: Applies the bondalign_23 feature map which:
   - Uses `RecenterBondLayer([1, 2], batch_mode=True)` to align atoms 1 and 2
   - Selects only carbon atoms (4 carbons × 3 coordinates = 12D input)
3. **DNet Forward Pass**: Maps the 12D feature space to a lower-dimensional manifold (typically 2-4D)
4. **Normal Computation**: Uses Open3D to compute pointwise normals on the manifold surface
5. **Visualization**: Creates plots showing the manifold points colored by dihedral angles and normal vectors

## Usage

```bash
cd learn/manifold_learning
python compute_normals_with_dnet.py
```

## Requirements

- PyTorch
- Open3D
- NumPy
- Matplotlib
- scikit-learn

## Input Files

- **Data**: `ground_data/butane_nonaligned.npz` - Butane molecular dynamics trajectory data
- **Model**: `outputs/bondalign_23/model_encoder_DNet_Laplacian_8oct2024_carbons` - Pre-trained DNet encoder model

## Output Files

The script saves results to `outputs/bondalign_23/`:
- `manifold_with_normals.npz` - NumPy archive containing:
  - `manifold_points`: Points on the learned manifold
  - `normals`: Computed normal vectors at each point
  - `dihedrals`: Original dihedral angles for visualization
  - `raw_data`: Original input data
- `manifold_normals_visualization.png` - Visualization plots

## Configuration

You can modify the following parameters in the `main()` function:
- `subsample_rate`: Controls data subsampling (default: 10)
- `radius`: Search radius for normal estimation (default: 0.1)
- `max_neighbors`: Maximum number of neighbors for normal estimation (default: 10)
- `batch_size`: Batch size for DNet inference (default: 1000)

## Technical Details

### Feature Map (bondalign_23)
The bondalign_23 feature map:
1. Applies bond alignment using atoms at indices [1, 2] (RecenterBondLayer)
2. Recenters and rotates the molecular configuration to align the bond
3. Works on carbon-only data (the input data already contains only 4 carbon atoms)
4. Returns a 12-dimensional feature vector (4 carbons × 3 coordinates)

### DNet Model
- **Input**: 12D (4 carbon atoms × 3 coordinates)
- **Output**: 3D manifold coordinates
- **Architecture**: `standard_4_layer_tanh_encoder_3D` wrapper around `standard_4_layer_tanh_encoder`
- **Base Model**: 4-layer neural network with Tanh activation functions
- **3D Wrapper**: Removes the last dimension to project to 3D manifold space

### Normal Computation
- Uses Open3D's `estimate_normals()` with hybrid search (radius + max neighbors)
- Search parameters: radius=0.1, max_neighbors=10
- Orients all normals consistently with respect to the center of mass of the point cloud
- For 2D manifolds, points are padded to 3D for Open3D compatibility
- Returns unit normal vectors at each manifold point

## Visualization

The script creates different visualizations depending on the manifold dimensionality:

**For 3D+ manifolds**:
- 3D scatter plot of manifold points colored by dihedral angles
- 3D plot showing normal vectors at subset of points
- Histogram of normal vector magnitudes

**For 2D manifolds**:
- 2D scatter plot colored by dihedral angles
- 2D plot with normal vectors
- Component-wise histograms of normal vectors
- Histogram of normal vector magnitudes 