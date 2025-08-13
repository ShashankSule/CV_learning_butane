from torch import nn
import torch

# Custom Matching Loss as a subclass of nn.Module
class MatchingLoss(nn.Module):
    def __init__(self):
        super(MatchingLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        """
        Computes the matching loss using Mean Squared Error (MSE).

        Parameters:
        - output: Predicted output from the model.
        - target: Ground truth target.

        Returns:
        - Matching loss value.
        """
        return self.mse_loss(output, target)

# Custom Laplacian Loss as a subclass of nn.Module
class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, laplacian, sigma, psi):
        """
        Computes the Laplacian loss.

        Parameters:
        - laplacian: Laplacian matrix.
        - sigma: Eigenvalues or scaling factor.
        - psi: Eigenvectors or transformed output.

        Returns:
        - Laplacian loss value.
        """
        Ltimes_psi = torch.matmul(laplacian, psi)
        sigma_times_psi = psi * sigma
        return self.mse_loss(Ltimes_psi, sigma_times_psi)