from typing import MappingView
import argparse
import torch
import os
from networks import FullyConnected

nets = os.listdir('./mnist_nets')

DEVICE='cpu'
INPUT_SIZE=28

for net_i in nets:


    if net_i.endswith('fc1.pt'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif net_i.endswith('fc2.pt'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif net_i.endswith('fc3.pt'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif net_i.endswith('fc4.pt'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif net_i.endswith('fc5.pt'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('./mnist_nets/%s' % net_i, map_location=torch.device(DEVICE)))
    print(net)