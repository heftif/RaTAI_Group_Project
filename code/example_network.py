import torch.nn as nn
import torch
from networks import Normalization
from networks import SPU

class ExampleNetwork(nn.Module):
    def __init__(self, device):
        super(ExampleNetwork, self).__init__()
        with torch.no_grad(): # we want to set fixed weights here, not optimize them
            layers = [Normalization(device), nn.Flatten()]
            input_size = 2

            # first affine layer
            affine1 = nn.Linear(input_size, input_size)
            affine1.weight = torch.nn.Parameter(torch.tensor([[-1.0, 2.1], [1.0, 1.0]]))
            affine1.bias = torch.nn.Parameter(torch.tensor([0.7, 0.0]))
            layers += [affine1]

            # SPU layer
            layers += [SPU()]

            # second affine layer
            affine2 = nn.Linear(input_size, input_size)
            affine2.weight = torch.nn.Parameter(torch.tensor([[0.5, 1.0], [-0.5, 1.0]]))
            affine2.bias = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
            layers += [affine2]
            
            self.layers = nn.Sequential(*layers)
            #print(self.layers)

    def forward(self, x):
        with torch.no_grad(): 
            return self.layers(x)

class ExampleNetwork_nocrossing(nn.Module):
    def __init__(self, device):
        super(ExampleNetwork_nocrossing, self).__init__()
        with torch.no_grad(): # we want to set fixed weights here, not optimize them
            layers = [Normalization(device), nn.Flatten()]
            input_size = 2

            # first affine layer
            affine1 = nn.Linear(input_size, input_size)
            affine1.weight = torch.nn.Parameter(torch.tensor([[-1.0, 1.5], [1.0, 1.0]]))
            affine1.bias = torch.nn.Parameter(torch.tensor([0.2, 0.0]))
            layers += [affine1]

            # SPU layer
            layers += [SPU()]

            # second affine layer
            affine2 = nn.Linear(input_size, input_size)
            affine2.weight = torch.nn.Parameter(torch.tensor([[0.5, -1.0], [-0.5, 2.0]]))
            affine2.bias = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
            layers += [affine2]

            self.layers = nn.Sequential(*layers)
            #print(self.layers)

    def forward(self, x):
        with torch.no_grad():
            return self.layers(x)
