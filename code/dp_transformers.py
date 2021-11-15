import torch.nn as nn
import torch
from networks import Normalization
from torch.autograd.functional import jacobian
from typing import Tuple

class DeepPolyInstance():
    def __init__(self, net, eps, inputs, true_label, steps_backsubstitution=0):
        self.net = net
        self.eps = eps
        self.inputs = inputs
        self.true_label = true_label
        self.steps_backsubstitution = steps_backsubstitution
        self.verifier = self._net_transform()

    def _net_transform(self):
        last = None
        layers=[DeepPolyInputNode(self.eps)]
        for layer in self.net.layers:
            if isinstance(layer, torch.nn.Linear):
                last = DeepPolyLinearNode(layer._parameters['weight'].detach(),layer._parameters['bias'].detach(), last=last, back_sub_steps=self.back_sub_steps)
                layers += [last]
            elif isinstance(layer, torch.nn.SPU):
                last = SPUTransformer(last=last, back_sub_steps=self.back_sub_steps, use_interpolation= self.use_interpolation)
                layers += [last]
            elif isinstance(layer, torch.nn.Flatten):
                last = FlattenTransformer(last=last)
                layers += [last]
            elif isinstance(layer, Normalization):
                last = DeepPolyNormalizingNode()
                layers += [last]
            else:
                raise TypeError("Layer not found")
        layers +=[PairwiseDifference(self.true_label, last=last, steps_backsubstitution=self.steps_backsubstitution)]
        return nn.Sequential(*layers)
        
    def verify(self):
        return self.verifier(self.inputs)


class DeepPolyInputNode(nn.Module):
    def __init__(self, eps):
        super(DeepPolyInputNode, self).__init__()
        self.eps = eps

    def forward(self, input):
        self.bounds = input.repeat(2, 1, 1, 1)
        self.bounds += torch.FloatTensor([[[[-self.eps]]], [[[self.eps]]]])
        self.bounds = torch.clamp(self.bounds, 0., 1.) # restric lower bound to 0 for the input since pixels go from 0 to 1
        return self.bounds


class DeepPolyNormalizingNode(nn.Module):
    def __init__(self, last=None):
        super(DeepPolyNormalizingNode, self).__init__()
        self.last = last
        self.mean = torch.FloatTensor([0.1307])
        self.sigma = torch.FloatTensor([0.3081])
    
    def forward(self, bounds):
        self.bounds = bounds
        self.bounds = torch.div(bounds - self.mean, self.sigma) # normalize bounds the same way the input is normalized (see networks.py --> Normalization class)
        return self.bounds


class DeepPolyLinearNode(nn.Module):
    def __init__(self, weights, bias=None, last=None, steps_backsubstitution=0):
        super(DeepPolyLinearNode, self).__init__()
        self.weights = weights
        self.bias = bias
        self.last = last
        self.steps_backsubstitution = steps_backsubstitution
        self.positive_weights = torch.clamp(self.weights, min=0.)
        self.negative_weights = torch.clamp(self.weights, max=0.)

    def forward(self, bounds):
        upper = torch.matmul(self.positive_weights, bounds[:,1]) + torch.matmul(self.negative_weights, bounds[:,0]) # bounds: alphas in Singh 2019?
        lower = torch.matmul(self.positive_weights, bounds[:,0]) + torch.matmul(self.negative_weights, bounds[:,1])
        self.bounds = torch.stack([lower, upper], 1)
        if self.bias is not None:
            self.bounds += self.bias.reshape(-1, 1) # add the bias where it exists
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds
    
    def back_sub(self, max_steps):
        new_bounds = self._back_sub(max_steps) # perform backsubstitution
        index_tighter_lower = new_bounds[:,0] > self.bounds[:,0] # find the bounds that can be tightened
        index_tighter_upper = new_bounds[:,1] < self.bounds[:,1]
        self.bounds[index_tighter_lower, 0] = new_bounds[index_tighter_lower,0] # update those bounds
        self.bounds[index_tighter_upper, 1] = new_bounds[index_tighter_upper,1]
        
    def _back_sub(self, max_steps, params : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None):
        if params is None:
            Ml, Mu, bl, bu = self.weights, self.weights, self.bias, self.bias
        else:
            Ml, Mu, bl, bu = params
        if max_steps > 0 and self.last.last.last is not None: #self.last.last.last is None happens only for the first two layers - these are always the Normalizing and the Flatten layers for all test nets
            Mlnew = torch.clamp(Ml, min=0) * self.last.beta + torch.clamp(Ml, max=0)* self.last.lmbda
            Munew = torch.clamp(Mu, min=0) * self.last.lmbda + torch.clamp(Mu, max=0)* self.last.beta
            blnew = bl + torch.matmul(torch.clamp(Ml, max=0), self.last.mu)
            bunew = bu + torch.matmul(torch.clamp(Mu, min=0), self.last.mu) 
            return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
        else:
            lower = torch.matmul(torch.clamp(Ml, min=0), self.last.bounds[:, 0]) + torch.matmul(torch.clamp(Ml, max=0), self.last.bounds[:, 1]) + bl
            upper = torch.matmul(torch.clamp(Mu, min=0), self.last.bounds[:, 1]) + torch.matmul(torch.clamp(Mu, max=0), self.last.bounds[:, 0]) + bu
            return torch.stack([lower, upper], 1)
