import torch.nn as nn
import torch

# write down architecture 
class periodic_activation(nn.Module):
    def __init__(self):
        super(periodic_activation, self).__init__()
    def forward(self,x): 
        return x + torch.sin(x)**2
        
class Encoder(nn.Module):
    def __init__(self, activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim):
        super(Encoder, self).__init__()
        
        # Defining the layers of the neural network
        # self.featurizer = feature_map
        self.activation = activation
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.hidden3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.hidden4 = nn.Linear(hidden3_dim, hidden4_dim)
        self.bottleneck = nn.Linear(hidden4_dim, encoder_dim)

        # Collecting layers for convenience
        self.encoder = nn.Sequential(self.hidden1, self.activation, \
                                     self.hidden2, self.activation, \
                                        self.hidden3, self.activation, \
                                            self.hidden4, self.activation, \
                                                self.bottleneck, self.activation)

    
    def encode(self, x):
        # y = self.featurizer(x)
        return self.encoder(x)
    
    # Required for any subclass of nn.module: defines how data passes through the `computational graph'
    def forward(self, x):
        # x = self.featurizer(x)
        return self.encode(x)
    
class Decoder(nn.Module):
    def __init__(self, activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim):
        super(Decoder, self).__init__()
        
        # Defining the layers of the neural network
        self.activation = activation
        self.hidden4 = nn.Linear(encoder_dim, hidden4_dim)
        self.hidden3 = nn.Linear(hidden4_dim, hidden3_dim)
        self.hidden2 = nn.Linear(hidden3_dim, hidden2_dim)
        self.hidden1 = nn.Linear(hidden2_dim, hidden1_dim)
        self.reconstruct = nn.Linear(hidden1_dim, input_dim)

        # Collecting layers for convenience as encoder and decoder
        self.decoder = nn.Sequential(self.hidden4, self.activation, self.hidden3, self.activation, self.hidden2, self.activation, self.hidden1, self.activation, self.reconstruct)

    
    def decode(self, z):
        return self.decoder(z)
        
    # Required for any subclass of nn.module: defines how data passes through the `computational graph'
    def forward(self, x):
        return self.decode(x)
    
# some standard encoders 
def standard_4_layer_dnet_tanh_encoder(input_dim, encoder_dim):
    hidden1_dim = 32
    hidden2_dim = 32
    hidden3_dim = 32
    hidden4_dim = 32
    activation = nn.Tanh()
    return Encoder(activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim)
def standard_4_layer_dnet_tanh_decoder(input_dim, encoder_dim):
    hidden1_dim = 32
    hidden2_dim = 32
    hidden3_dim = 32
    hidden4_dim = 32
    activation = nn.Tanh()
    return Decoder(activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim)
def standard_4_layer_dnet_snake_encoder(input_dim, encoder_dim):
    hidden1_dim = 32
    hidden2_dim = 32
    hidden3_dim = 32
    hidden4_dim = 32
    activation = periodic_activation()
    return Encoder(activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim)
def standard_4_layer_dnet_snake_decoder(input_dim, encoder_dim):
    hidden1_dim = 32
    hidden2_dim = 32
    hidden3_dim = 32
    hidden4_dim = 32
    activation = periodic_activation()
    return Decoder(activation, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, encoder_dim)

class standard_4_layer_dnet_snake_encoder_3D(nn.Module):
    def __init__(self, model):
        super(standard_4_layer_dnet_snake_encoder_3D, self).__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        return y[..., :-1]

class standard_4_layer_dnet_tanh_encoder_3D(nn.Module):
    def __init__(self, model):
        super(standard_4_layer_dnet_tanh_encoder_3D, self).__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        return y[..., :-1]