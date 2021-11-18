import torch.nn as nn
import torch
from networks import Normalization
from networks import SPU
from torch.autograd.functional import jacobian
from typing import Tuple

class DeepPolyInstance():
    def __init__(self, net, eps, inputs, true_label, steps_backsub=0):
        self.net = net
        self.eps = eps
        self.inputs = inputs
        self.true_label = true_label
        self.steps_backsub = steps_backsub

    def verifier_net(self):
        last = None
        layers = [InputNode(self.eps)]
        for layer in self.net.layers:
            if isinstance(layer, Normalization):
                last = NormalizingNode()
                layers += [last]
            elif isinstance(layer, torch.nn.Flatten):
                last = FlattenTransformer(last=last)
                layers += [last]
            elif isinstance(layer, torch.nn.Linear):
                last = LinearTransformer(layer._parameters['weight'].detach(), layer._parameters['bias'].detach(), 
                        last=last, steps_backsub=self.steps_backsub)
                layers += [last]
            elif isinstance(layer, SPU):
                last = SPUTransformer(self.inputs, last=last, steps_backsub=self.steps_backsub)
                layers += [last]
            else:
                raise TypeError("Layer not found")
        layers +=[PairwiseDifference(self.true_label, last=last, steps_backsub=self.steps_backsub)]
        return nn.Sequential(*layers)


class InputNode(nn.Module):
    def __init__(self, eps):
        super(InputNode, self).__init__()
        self.eps = eps

    def forward(self, input):
        self.bounds = input.repeat(2, 1, 1, 1)
        self.bounds += torch.FloatTensor([[[[-self.eps]]], [[[self.eps]]]])
        self.bounds = torch.clamp(self.bounds, 0., 1.) # restrict lower bound to 0 for the input since pixels go from 0 to 1
        return self.bounds


class NormalizingNode(nn.Module):
    def __init__(self, last=None):
        super(NormalizingNode, self).__init__()
        self.last = last
        self.mean = torch.FloatTensor([0.1307])
        self.sigma = torch.FloatTensor([0.3081])
    
    def forward(self, bounds):
        self.bounds = bounds
        self.bounds = torch.div(bounds - self.mean, self.sigma) # normalize bounds the same way the input is normalized (see networks.py --> Normalization class)
        return self.bounds

class FlattenTransformer(nn.Module):
    def __init__(self, last=None):
        super(FlattenTransformer, self).__init__()
        self.last = last

    def forward(self, bounds):
        return torch.stack([bounds[0,:,:,:].flatten(), bounds[1,:,:,:].flatten()], 1)

    def back_sub(self):
        # TODO: do we even need this?
        pass


class LinearTransformer(nn.Module):
    def __init__(self, weights, bias=None, last=None, steps_backsub=0):
        super(LinearTransformer, self).__init__()
        self.weights = weights
        self.bias = bias
        self.last = last
        self.steps_backsub = steps_backsub
        self.positive_weights = torch.clamp(self.weights, min=0)
        self.negative_weights = torch.clamp(self.weights, max=0)

    def forward(self, bounds):
        lower = torch.matmul(self.positive_weights, bounds[:,0]) + torch.matmul(self.negative_weights, bounds[:,1]) # bounds[:,0] are all lower bounds, bounds[:,1] are all upper bounds
        upper = torch.matmul(self.positive_weights, bounds[:,1]) + torch.matmul(self.negative_weights, bounds[:,0]) 
        self.bounds = torch.stack([lower, upper], 1)
        if self.bias is not None:
            self.bounds += self.bias.reshape(-1, 1) # add the bias where it exists
        if self.steps_backsub > 0:
            self.back_sub()
        return self.bounds
    
    def back_sub(self):  
        # TODO: back substitution overall 
        # how to back sub through SPU and affine layers?
        # 
        #      
        # we want to update our linear node's weights and biases using the last node's weights, biases and bounds
        steps = self.steps_backsub
        current_node = self 

        while steps > 0 and current_node.last.last.last is not None: # self.last.last.last is only None for the first two layers - these are always the Normalizing and the Flatten layers for all test nets
            weights_new = torch.matmul(weights, current_node.last.weights)
            bias_new = torch.matmul(weights, current_node.last.bias) + bias

            weights = weights_new
            bias = bias_new
            current_node = current_node.last # move back another layer
            steps -= 1

        # compute new lower and upper bounds by inserting the bounds at current_node
        positive_weights = torch.clamp(weights, min=0)
        negative_weights = torch.clamp(weights, max=0)
        lower = torch.matmul(positive_weights, current_node.bounds[:,0]) + \
                torch.matmul(negative_weights, current_node.bounds[:,1]) + bias 
        upper = torch.matmul(positive_weights, current_node.bounds[:,1]) + \
                torch.matmul(negative_weights, current_node.bounds[:,0]) + bias

        # find valid new bounds and set these as new bounds
        valid_lower = lower > self.bounds[:,0] 
        valid_upper = upper < self.bounds[:,1]
        self.bounds[valid_lower, 0] = lower[valid_lower] 
        self.bounds[valid_upper, 1] = upper[valid_upper]


class SPUTransformer(nn.Module):
    def __init__(self, inputs, last=None, steps_backsub=0):
        super(SPUTransformer, self).__init__()
        self.inputs = inputs
        self.last = last
        self.steps_backsub = steps_backsub

    def spu(x):
        ''' SPU function '''
        return torch.where(x > 0, x**2 - 0.5, torch.sigmoid(-x) - 1)

    def forward(self, bounds):
        ''' update bounds according to SPU function '''
        ### case 1: interval is non-positive
        # value range of sigmoid(-x) - 1 is [-0.5, 0] for x <= 0
        # lower line: constant at -0.5
        # upper line: constant at 0
        neg_ind = bounds[:,1]<=0
        bounds[neg_ind,0] = torch.full_like(bounds[neg_ind,0], -0.5)
        bounds[neg_ind,1] = torch.full_like(bounds[neg_ind,1], 0.0)
        
        ### case 2: interval is non-negative
        # lower line: constant at -0.5
        # upper line: use line between SPU(l) and SPU(u)
        pos_ind = bounds[:,0]>=0
        print(bounds[pos_ind,1])
        slopes = (self.spu(bounds[pos_ind,1]) - self.spu(bounds[pos_ind,0])) \
                / (bounds[pos_ind,1] - bounds[pos_ind,0])
        intercepts = torch.square(self.inputs[pos_ind]) - slopes*self.inputs[pos_ind] \
                - torch.full_like(bounds[pos_ind,0], 0.5)
        bounds[pos_ind,0] = torch.full_like(bounds[neg_ind,0], -0.5)
        bounds[pos_ind,1] = slopes*self.inputs[pos_ind] + intercepts

        ### case 3: interval crosses 0
        # lower line: constant at -0.5
        # upper line: use line between SPU(l) and SPU(u) as in case 2
        cross_ind = torch.logical_not(torch.logical_or(neg_ind, pos_ind)) # find remaining indices
        slopes = (self.spu(bounds[cross_ind,1]) - self.spu(bounds[cross_ind,0])) \
                / (bounds[cross_ind,1] - bounds[cross_ind,0])
        intercepts = torch.square(self.inputs[cross_ind]) - slopes*self.inputs[cross_ind] \
                - torch.full_like(bounds[cross_ind,0], 0.5)
        bounds[cross_ind,0] = torch.full_like(bounds[neg_ind,0], -0.5)
        bounds[cross_ind,1] = slopes*self.inputs[cross_ind] + intercepts

        # use backsubstitution in case it is requested
        if self.steps_backsub > 0:
            self.back_sub()

    def back_sub(self):
        # TODO: implement
        pass

class BoxTransformer(nn.Module):
    ''' simple Box Transformer for SPU function '''
    def __init__(self, inputs, last=None, steps_backsub=0):
        super(BoxTransformer, self).__init__()
        self.inputs = inputs
        self.last = last
        self.steps_backsub = steps_backsub

    def spu(x):
        ''' SPU function '''
        return torch.where(x > 0, x**2 - 0.5, torch.sigmoid(-x) - 1)

    def forward(self, bounds):
        bounds[:,0] = torch.min(self.spu(bounds[:,0]), self.spu(bounds[:,1]))
        bounds[:,1] = torch.max(self.spu(bounds[:,0]), self.spu(bounds[:,1]))

        # use backsubstitution in case it is requested
        if self.steps_backsub > 0:
            self.back_sub()

    def back_sub(self):
        # TODO: implement
        pass


class PairwiseDifference(nn.Module):
    def __init__(self, true_label, last=None, steps_backsub=0):
        super(PairwiseDifference, self).__init__()
        self.last = last
        self.true_label = true_label
        self.steps_backsub = steps_backsub
        # TODO: finish initialization

    def forward(self, bounds):
        # TODO: implement
        pass

    def back_sub(self, steps_backsub):
        # TODO: implement
        pass

