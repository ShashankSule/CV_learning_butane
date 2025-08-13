import torch.nn as nn
import torch

class periodic_activation(nn.Module):
    def __init__(self):
        super(periodic_activation, self).__init__()
    def forward(self,x): 
        return x + torch.sin(x)**2

class PsiNetwork(nn.Module):
    def __init__(self, activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim):
        super(PsiNetwork, self).__init__()
        
        # Defining the layers of the neural network
        self.activation = activation
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.hidden3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.hidden4 = nn.Linear(hidden3_dim, hidden4_dim)
        self.bottleneck = nn.Linear(hidden4_dim, encoder_dim)

        self.Psif = nn.Sequential(
            self.hidden1, self.activation, 
            self.hidden2, self.activation,
            self.hidden3, self.activation, 
            self.hidden4, self.activation,
            self.bottleneck
        )

    def Psi(self, x):
        return self.Psif(x)
    
    def forward(self, x):
        return self.Psi(x)

def standard_4_layer_potential_net(input_dim, output_dim=1):
    hidden1_dim = 30
    hidden2_dim = 45
    hidden3_dim = 32
    hidden4_dim = 32
    activation = periodic_activation()
    return PsiNetwork(activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, output_dim) 