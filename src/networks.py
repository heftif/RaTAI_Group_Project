from typing import ForwardRef
import torch
import torch.nn as nn


class Normalization(nn.Module):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class SPU(nn.Module):

    def forward(self, x):
        return torch.where(x > 0, x**2 - 0.5, torch.sigmoid(-x) - 1)

class derivative_spu(nn.Module):

    def forward(self, x):
        return torch.where(x > 0, 2*x, torch.div(-1*torch.exp(x), torch.square(torch.add(torch.exp(x), torch.ones_like(x)))))


class FullyConnected(nn.Module):

    def __init__(self, device, input_size, fc_layers):
        super(FullyConnected, self).__init__()

        layers = [Normalization(device), nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [SPU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


