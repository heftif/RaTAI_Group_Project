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


    #building the deeppoly net according to the given net structure
    def verifier_net(self):
        last = None
        #get upper and lower bounds of the inputs
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
        layers +=[VerifyRobustness(self.true_label, last=last, steps_backsub=self.steps_backsub)]
        return nn.Sequential(*layers)


#calculating the upper and lower bounds from the input tensor
class InputNode(nn.Module):
    def __init__(self, eps):
        super(InputNode, self).__init__()
        self.eps = eps

    def forward(self, input):
        if input.dim() == 4: # this is the input we expect
            self.bounds = input.repeat(2, 1, 1, 1)
            self.bounds += torch.FloatTensor([[[[-self.eps]]], [[[self.eps]]]])
            self.bounds = torch.clamp(self.bounds, min=0., max=1.) # restrict lower bound to 0 for the input since pixels go from 0 to 1

        else: # e.g. for test input 
            self.bounds = input.repeat(1, 2)
            self.bounds += torch.FloatTensor([-self.eps, self.eps])
            self.bounds = torch.clamp(self.bounds, min=0., max=1.) # restrict lower bound to 0 for the input since pixels go from 0 to 1
            print(f"INPUT:\n{input}\nINITIAL BOUNDS:\n{self.bounds}\n=====================================")
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
        print(f"BOUNDS AFTER NORMALIZING LAYER:\n{self.bounds}\n=====================================")
        return self.bounds

class FlattenTransformer(nn.Module):
    def __init__(self, last=None):
        super(FlattenTransformer, self).__init__()
        self.last = last

    def forward(self, bounds):
        if bounds.dim() == 4:
            self.bounds = torch.stack([bounds[0,:,:,:].flatten(), bounds[1,:,:,:].flatten()], 1)
            return self.bounds
        else: # e.g. for test input
            self.bounds = bounds
            return bounds


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
            
        print(f"BOUNDS AFTER AFFINE LAYER:\n{self.bounds}\n=====================================")
        return self.bounds

    def back_sub(self, steps, lower_slopes, upper_slopes, shift, lower_bias, upper_bias):
        if steps > 0:
            backsub_bounds = self.back_sub_from_top_layer(steps, lower_slopes, upper_slopes, shift, lower_bias, upper_bias)
            # check that the new bounds are valid
            valid_lower = backsub_bounds[:,0] > self.bounds[:,0]
            valid_upper = backsub_bounds[:,1] < self.bounds[:,1]
            self.bounds[valid_lower, 0] = backsub_bounds[:,0][valid_lower]
            self.bounds[valid_upper, 1] = backsub_bounds[:,1][valid_upper]

        return self.bounds



    def back_sub_from_top_layer(self, steps, lower_slopes, upper_slopes, shift, lower_bias, upper_bias):
        current_node = self
            
        if steps > 0 and current_node.last.last.last is not None:
            #TODO: check if everything here works and if it yields the expected results
            #do the backpropagation again, get the layers together
            #we always backstep from an SPU layer and always step back into an SPU layer (I think?)
            #slope_i*weights(i-2)*bounds[:,0](i-2) + slope_i*bias_(i-2) + shift
            #and now we need to replace the bounds again -> insert the spu bounds:
            #bounds[:,0] = slope*bounds[] + shift
            #slope_i*weights*(slope*bounds + shift) + slope_i*bias + shift
            #looking for the lowest and highest values is only done at the very end, as we can not tell from
            #here on, which will yield the lowest or highest bounds.
            # lower_bias = lower_slopes * self.bias
            # lower_slopes = lower_slopes*self.weights
            # upper_bias = upper_slopes * self.bias
            # upper_slopes = upper_slopes*self.weights

            lower_bias = lower_bias + torch.matmul(torch.clamp(lower_slopes, max=0), self.last.shift)
            upper_bias = upper_bias + torch.matmul(torch.clamp(upper_slopes, min=0), self.last.shift)
            lower_slopes = lower_slopes*self.weights
            upper_slopes = upper_slopes*self.weights

            return self.last.backsub(steps-1, lower_slopes, upper_slopes, self.shift, lower_bias, upper_bias)

        else:
            #we are finished with backstepping and thus need to substitute the values in
            positive_weights = torch.clamp(self.weights, min=0)
            negative_weights = torch.clamp(self.weights, max=0)
            #now calculate: slope_i*weights(i-2)*bounds[:,0](i-2) + slope_i*bias_(i-2) + shift
            #TODO: fix the dimensions and use matmul, so the code works
            # lower = lower_slopes*positive_weights*current_node.bounds[:,0] + \
            #         lower_slopes*negative_weights*current_node.bounds[:,1] + \
            #         lower_slopes*current_node.bias #+ shift
            # upper = upper_slopes*positive_weights*current_node.bounds[:,1] + \
            #         upper_slopes*negative_weights*current_node.bounds[:,0] + \
            #         upper_slopes*current_node.bias #+ shift
            lower = torch.matmul(torch.clamp(lower_slopes, min=0), self.last.bounds[:,0]) + torch.matmul(torch.clamp(upper_slopes, max=0), self.last.bounds[:,1])
            upper = torch.matmul(torch.clamp(upper_slopes, min=0), self.last.bounds[:, 1]) + torch.matmul(torch.clamp(lower_slopes, max=0), self.last.bounds[:, 0])
            return torch.stack([lower,upper],1)
    

class SPUTransformer(nn.Module):
    def __init__(self, last=None, steps_backsub=0):
        super(SPUTransformer, self).__init__()
        #self.inputs = inputs.flatten()
        self.last = last
        self.steps_backsub = steps_backsub
        # self.shift
        # self.slope
        # self.crossing

    def forward(self, bounds):
        ''' update bounds according to SPU function '''
        spu = SPU()
        #initialise shift, slopes and param
        self.slopes = torch.zeros_like(bounds[:, 1])
        self.shift = torch.zeros_like(bounds[:, 1])
        self.negative = torch.zeros_like(bounds[:, 1])
        self.positive = torch.zeros_like(bounds[:, 1])
        self.crossing = torch.zeros_like(bounds[:, 1])


        #calculate the spu values of all bounds
        val_spu = torch.zeros_like(bounds)
        val_spu[:,0] = spu(bounds[:,0])
        val_spu[:,1] = spu(bounds[:,1])

        #calculate the difference
        diff = (bounds[:,1] - bounds[:,0])
        #calculate the slopes
        self.slopes = torch.div(val_spu[:,1]-val_spu[:,0], diff)
        #we have all the slopes, we just need to be careful that the calculations are sound
        #when we deal with crossing values! -> lower bounds must be -0.5 in this case for the approximation!
        #maybe just use a box in this case? Either box or use a triangle that's extended with the given slope
        #down to -0.5
        #if slope is negative and non crossing => lower bound of sigmoid part
        #if slope is positive and non crossing => upper bound of parabola part

        #calculate the shift (to get the full linear description (y=slope*x + shift))
        self.shift = val_spu[:,1] - self.slopes*bounds[:,1]
        ### case 1: interval is non-positive
        neg_ind = bounds[:,1]<=0
        self.neg_ind = neg_ind
        ### case 2: interval is non-negative
        pos_ind = bounds[:,0]>=0
        self.pos_ind = pos_ind
        ### case 3: crossing
        cross_ind = torch.logical_not(torch.logical_or(neg_ind, pos_ind))
        self.cross_ind = cross_ind

        #calculate the new bounds -> just take the function value. These are just l and u, the inequalities
        #are only relevant in the back substitution!
        self.bounds = torch.zeros_like(bounds)
        lower_negative_bounds = spu(bounds[neg_ind,1]) # for the only negative case the bounds are inverted - see graph
        lower_positive_bounds = spu(bounds[pos_ind,0])
        upper_negative_bounds = spu(bounds[neg_ind,0])
        upper_positive_bounds = spu(bounds[pos_ind,1])
        lower_crossing_bounds = -0.5
        upper_crossing_bounds = torch.max(spu(bounds[cross_ind,0]), spu(bounds[cross_ind,1]))
        self.bounds = spu(bounds) 
        self.bounds[cross_ind,0] = lower_crossing_bounds
        self.bounds[cross_ind,1] = upper_crossing_bounds
        self.bounds[neg_ind,0] = lower_negative_bounds
        self.bounds[neg_ind,1] = upper_negative_bounds
        self.bounds[pos_ind,0] = lower_positive_bounds
        self.bounds[pos_ind,1] = upper_positive_bounds

        # set bounds of crossing indexes to -0.5 and max(spu(l), spu(u)) - done above (Angelos)

        print(f"BOUNDS AFTER SPU LAYER:\n{self.bounds}\n=====================================")
        return self.bounds

    #for when we do the first backsubstitution
    def back_sub(self, steps):
        #calculate the boundaries with the correct approximation
        #from the paper: perform back substitution as matrix multiplication.
        if steps > 0:
            #the equation of the linear boundary we set (upper or lower) is:
            #x_i = lambda * x_(i-1) + shift
            #in backsubstitution, we want to push the bounds from the layer before through our calculated approximation
            #thus: [lower_new_i] = slope_i * [lower_(i-1)] + shift_i
            #lower_(i-1) = weights_(i-2) * bounds[:,0](i-2) + bias_(i-2)
            #=> [lower_new_i] = slope_i * [weights_(i-2) * bounds[:,0](i-2) + bias_(i-2)] + shift_i
            #this is equal to:  slope_i*weights(i-2)*bounds[:,0](i-2) + slope_i*bias_(i-2) + shift
            #for purely negative intervals: lower is calculated as slope, upper is fixed (function value)
            #for purely positive intervals: upper is calculated as slope, lower is fixed (function value)
            #for crossing intervals: box, with lower = -0.5 and upper = max(spu(li),spu(ui)) ->
            #later, I would suggest extending the slope line we already have to -0.5, if this area is smaller than te box area

            #thus, we need to hand down the matrixes and information we have in this object only, to the next layer down
            #and there either hand it back farther (by substituting the information we need from that layer) or insert
            #the bounds and give the better approximation back.

            #what about the linear bounds, as we approximate with a triangle and either the upper or lower bound is then
            #of the same equation form, but with lamdba = 0 => lower_new_i = shift.
            #thus follow the following calculated bounds:

            lower_slopes = self.slopes
            upper_slopes = self.slopes
            lower_slopes[self.pos_ind] = 0
            #box
            lower_slopes[self.cross_ind] = 0
            upper_slopes[self.neg_ind] = 0
            #box
            upper_slopes[self.cross_ind] = 0

            lower_bias = torch.zeros_like(lower_slopes)
            upper_bias = torch.zeros_like(upper_slopes)

            return self.last.back_sub(steps-1, lower_slopes, upper_slopes, self.shift, lower_bias, upper_bias)

    #overloading the function if we come from a layer higher up.
    def back_sub_from_top_layer(self, steps, lower_slopes, upper_slopes, shift, lower_bias, upper_bias):
        current_node = self
        #slope_i*weights*(slope*bounds + shift) + slope_i*bias + shift
        if steps > 0 and current_node.last.last.last is not None:
            #TODO: continue with the implementation.
            #but we need to again set the lower_slopes = 0 of all positive slopes etc.
            lower_slopes = lower_slopes * self.slopes
        else:
            #we are finished with backstepping and thus need to substitute the values in
            positive_weights = torch.clamp(self.weights, min=0)
            negative_weights = torch.clamp(self.weights, max=0)
            #now calculate: slope_i*weights(i-2)*bounds[:,0](i-2) + slope_i*bias_(i-2) + shift
            #TODO: fix the dimensions and use matmul, so the code works
            #TODO: we have obviously no weight in this layer => this should be with slopes and shifts
            lower = lower_slopes*positive_weights*current_node.bounds[:,0] + \
                    lower_slopes*negative_weights*current_node.bounds[:,1] + \
                    lower_slopes*current_node.bias + shift
            upper = upper_slopes*positive_weights*current_node.bounds[:,1] + \
                    upper_slopes*negative_weights*current_node.bounds[:,0] + \
                    upper_slopes*current_node.bias + shift
            return torch.stack([lower,upper],1)




class SPUTransformer_Test(nn.Module):
    def __init__(self, inputs, last=None, steps_backsub=0):
        super(SPUTransformer_Test, self).__init__()
        #self.inputs = inputs.flatten()
        self.last = last
        self.steps_backsub = steps_backsub

    def forward(self, bounds):
        ''' update bounds according to SPU function '''
        spu = SPU()
        ### case 1: interval is non-positive
        # value range of sigmoid(-x) - 1 is [-0.5, 0] for x <= 0
        # lower bound: constant at given upper bound
        # upper bound: constant at 0
        #TODO: lower line should be constant to the upper bound, not to -0.5
        neg_ind = bounds[:,1]<=0
        #set lower bound
        bounds[neg_ind,0] = torch.full_like(bounds[neg_ind,0], -0.5)
        #set upper bound
        bounds[neg_ind,1] = torch.full_like(bounds[neg_ind,1], 0.0)
        
        ### case 2: interval is non-negative
        # lower line: constant at -0.5
        # upper line: use line between SPU(l) and SPU(u)
        #TODO: lower line should be constant to lower bound, not to -0.5
        pos_ind = bounds[:,0]>=0
        val_spu = torch.zeros_like(bounds[pos_ind])
        val_spu[:,0] = spu(bounds[pos_ind,0])
        val_spu[:,1] = spu(bounds[pos_ind,1])
        diff = (bounds[pos_ind,1] - bounds[pos_ind,0])
        #calculate the slopes of the fully positiv valued
        slopes = torch.div(val_spu[:,1]-val_spu[:,0], diff)
        #calculate the
        intercepts = torch.square(self.inputs[pos_ind]) - slopes*self.inputs[pos_ind] - torch.full_like(bounds[pos_ind,0], 0.5)
        bounds[pos_ind,0] = torch.full_like(bounds[neg_ind,0], -0.5)
        bounds[pos_ind,1] = slopes*self.inputs[pos_ind] + intercepts

        ### case 3: interval crosses 0
        # lower line: constant at -0.5
        # upper line: use line between SPU(l) and SPU(u) as in case 2
        cross_ind = torch.logical_not(torch.logical_or(neg_ind, pos_ind)) # find remaining indices
        slopes = (spu(bounds[cross_ind,1]) - spu(bounds[cross_ind,0])) \
                / (bounds[cross_ind,1] - bounds[cross_ind,0])
        intercepts = torch.square(self.inputs[cross_ind]) - slopes*self.inputs[cross_ind] \
                - torch.full_like(bounds[cross_ind,0], 0.5)
        bounds[cross_ind,0] = torch.full_like(bounds[neg_ind,0], -0.5)
        bounds[cross_ind,1] = slopes*self.inputs[cross_ind] + intercepts

        # use backsubstitution in case it is requested
        if self.steps_backsub > 0:
            self.back_sub(self.steps_backsub)

    def back_sub(self, steps):

        pass

class BoxTransformer(nn.Module):
    ''' simple Box Transformer for SPU function '''
    def __init__(self, inputs, last=None, steps_backsub=0):
        super(BoxTransformer, self).__init__()
        self.inputs = inputs
        self.last = last
        self.steps_backsub = steps_backsub

    def forward(self, bounds):
        spu = SPU()
        bounds[:,0] = torch.min(spu(bounds[:,0]), spu(bounds[:,1]))
        bounds[:,1] = torch.max(spu(bounds[:,0]), spu(bounds[:,1]))

        # use backsubstitution in case it is requested
        if self.steps_backsub > 0:
            self.back_sub()

    def back_sub(self):
        # TODO: implement
        pass


class VerifyRobustness(nn.Module):
    def __init__(self, true_label, last=None, steps_backsub=0):
        super(VerifyRobustness, self).__init__()
        self.last = last
        self.true_label = true_label
        self.steps_backsub = steps_backsub
        # TODO: finish initialization

    def forward(self, bounds):
        # TODO: implement
        pass

    def back_sub(self, steps_backsub):
        # TODO: implement
        # call last.back_sub here
        # -> moves some layers back through more last.back_sub calls and then passes results to this layer again
        pass

