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
                last = SPUTransformer(last=last, steps_backsub=self.steps_backsub)
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

        print(f"BOUNDS AFTER AFFINE LAYER, before Backsub:\n{self.bounds}\n=====================================")

        if (self.steps_backsub > 0) and self.last.last.last is not None: # no backsub needed for the first affine layer
            backsub_bounds = self.back_sub(self.steps_backsub)

            valid_lower = backsub_bounds[:,0] > self.bounds[:,0]
            valid_upper = backsub_bounds[:,1] < self.bounds[:,1]
            self.bounds[valid_lower, 0] = backsub_bounds[:,0][valid_lower]
            self.bounds[valid_upper, 1] = backsub_bounds[:,1][valid_upper]

        print(f"BOUNDS AFTER AFFINE LAYER:\n{self.bounds}\n=====================================")
        return self.bounds

    def back_sub(self, steps):
        current_node = self
        if steps > 0:
            upper_Matrix = self.weights
            lower_Matrix = self.weights

            upper_Vector = self.bias
            lower_Vector = self.bias

            return self.last.back_sub_from_top_layer(steps-1, upper_Matrix, lower_Matrix, upper_Vector, lower_Vector)

        else:
            return self.bounds

    def back_sub_from_top_layer(self, steps, upper_Matrix, lower_Matrix, upper_Vector, lower_Vector):
        current_node = self
        Upper_Boundary_Matrix = torch.matmul(upper_Matrix, self.weights)
        Upper_Boundary_Vector = torch.matmul(upper_Matrix, self.bias) + upper_Vector
        Lower_Boundary_Matrix = torch.matmul(lower_Matrix, self.weights)
        Lower_Boundary_Vector = torch.matmul(lower_Matrix, self.bias) + lower_Vector

        #print(f"Upper Boundary Matrix:\n{Upper_Boundary_Matrix}\n=====================================")
        #print(f"Upper Boundary Vector:\n{Upper_Boundary_Vector}\n=====================================")
        #print(f"Lower Boundary Matrix:\n{Lower_Boundary_Matrix}\n=====================================")
        #print(f"Lower Boundary Vector:\n{Lower_Boundary_Vector}\n=====================================")

        if steps > 0 and current_node.last.last.last is not None:
            return self.last.back_sub_from_top_layer(steps-1, Upper_Boundary_Matrix, Lower_Boundary_Matrix, Upper_Boundary_Vector, Lower_Boundary_Vector)

        else:
            Upper_Boundary_Pos = torch.clamp(Upper_Boundary_Matrix, min=0)
            Upper_Boundary_Neg = torch.clamp(Upper_Boundary_Matrix, max=0)
            Lower_Boundary_Pos = torch.clamp(Lower_Boundary_Matrix, min=0)
            Lower_Boundary_Neg = torch.clamp(Lower_Boundary_Matrix, max=0)

            lower = torch.matmul(Lower_Boundary_Pos, self.last.bounds[:,0]) \
                    + torch.matmul(Lower_Boundary_Neg, self.last.bounds[:,1]) + Lower_Boundary_Vector
            upper = torch.matmul(Upper_Boundary_Pos, self.last.bounds[:, 1]) \
                    + torch.matmul(Upper_Boundary_Neg, self.last.bounds[:, 0]) + Upper_Boundary_Vector

            #print(f"lower:\n{lower}\n=====================================")
            #print(f"upper:\n{upper}\n=====================================")

            return torch.stack([lower,upper],1)

class SPUTransformer(nn.Module):
    def __init__(self, last=None, steps_backsub=0):
        super(SPUTransformer, self).__init__()
        self.last = last
        self.steps_backsub = steps_backsub
        # self.shift
        # self.slope
        # self.crossing

    def forward(self, bounds):
        ''' update bounds according to SPU function '''
        spu = SPU()
        #initialise shift, slopes and param
        self.slopes = torch.zeros_like(bounds)
        self.shifts = torch.zeros_like(bounds)

        #calculate the spu values of all bounds
        val_spu = torch.zeros_like(bounds)
        val_spu[:,0] = spu(bounds[:,0])
        val_spu[:,1] = spu(bounds[:,1])

        #calculate the difference
        diff = (bounds[:,1] - bounds[:,0])

        ### case 1: interval is non-positive
        neg_ind = bounds[:,1]<=0
        self.neg_ind = neg_ind
        #self.negative[neg_ind] = 1
        ### case 2: interval is non-negative
        pos_ind = bounds[:,0]>=0
        self.pos_ind = pos_ind
        #self.positive[pos_ind] = 1
        ### case 3: crossing
        cross_ind = torch.logical_not(torch.logical_or(neg_ind, pos_ind))
        self.cross_ind = cross_ind

        all_slopes = torch.div(val_spu[:,1]-val_spu[:,0], diff)

        #calculate the upper slopes (for crossing and purely positive intervals, for the others it's 0=>constant)
        self.slopes[pos_ind,1] = all_slopes[pos_ind]
        self.slopes[cross_ind,1] = all_slopes[cross_ind]
        #calculate the lower slopes (for purely negative intervals, for the others it's 0=>constant)
        self.slopes[neg_ind,0] = all_slopes[neg_ind]

        #calculate the shifts (to get the full linear description (y=slope*x + shift))
        self.shifts[pos_ind,1] = val_spu[pos_ind,1] - self.slopes[pos_ind,1]*bounds[pos_ind,1]
        self.shifts[cross_ind,1] = val_spu[cross_ind,1] - self.slopes[cross_ind,1]*bounds[cross_ind,1]
        self.shifts[neg_ind,0] = val_spu[neg_ind,1] - self.slopes[neg_ind,0]*bounds[neg_ind,1]

        print(f"SLOPES in SPU:\n{self.slopes}\n=====================================")
        print(f"SHIFTS in SPU, before fixing:\n{self.shifts}\n=====================================")

        #calculate the new bounds -> just take the function value. These are just l and u, the inequalities
        #are only relevant in the back substitution!
        self.bounds = spu(bounds)

        #change the upper bounds to the lower bound for all negative slopes (for crossing & purely negative)
        newupper = self.bounds[all_slopes < 0, 0]
        newlower = self.bounds[all_slopes < 0, 1]
        self.bounds[all_slopes < 0, 0] = newlower
        self.bounds[all_slopes < 0, 1] = newupper

        #set the lower bounds of crossing indexes to -0.5
        self.bounds[self.cross_ind,0] = -0.5

        #set the shifts of the remaining values constant to their upper/lower bounds
        self.shifts[neg_ind,1] = self.bounds[neg_ind,1]
        self.shifts[pos_ind,0] = self.bounds[pos_ind,0]
        self.shifts[cross_ind,0] = self.bounds[cross_ind,0]

        print(f"BOUNDS in SPU, before Backsub:\n{self.bounds}\n=====================================")
        print(f"SHIFTS in SPU, after fixing:\n{self.shifts}\n=====================================")

        #set bounds of crossing indexes to -0.5 and
        #self.bounds = torch.zeros_like(bounds)
        #lower_negative_bounds = spu(bounds[neg_ind,1]) # for the only negative case the bounds are inverted - see graph
        #lower_positive_bounds = spu(bounds[pos_ind,0])
        #upper_negative_bounds = spu(bounds[neg_ind,0])
        #upper_positive_bounds = spu(bounds[pos_ind,1])
        #lower_crossing_bounds = -0.5
        #upper_crossing_bounds = torch.max(spu(bounds[cross_ind,0]), spu(bounds[cross_ind,1]))
        #self.bounds = spu(bounds)
        #self.bounds[cross_ind,0] = lower_crossing_bounds
        #self.bounds[cross_ind,1] = upper_crossing_bounds
        #self.bounds[neg_ind,0] = lower_negative_bounds
        #self.bounds[neg_ind,1] = upper_negative_bounds
        #self.bounds[pos_ind,0] = lower_positive_bounds
        #self.bounds[pos_ind,1] = upper_positive_bounds



        # use backsubstitution in case it is requested
        if self.steps_backsub > 0:
            backsub_bounds = self.back_sub(self.steps_backsub)

            #check if the bounds are better then the old bounds
            valid_lower = backsub_bounds[:,0] > self.bounds[:,0]
            valid_upper = backsub_bounds[:,1] < self.bounds[:,1]
            self.bounds[valid_lower, 0] = backsub_bounds[:,0][valid_lower]
            self.bounds[valid_upper, 1] = backsub_bounds[:,1][valid_upper]

        print(f"BOUNDS after SPU, with backsub:\n{self.bounds}\n=====================================")
        return self.bounds

    #for when we do the first backsubstitution
    def back_sub(self, steps):
        current_node = self
        if steps > 0 and current_node.last.last is not None:
            upper_Matrix = torch.diag(self.slopes[:,1]) #diagonal upper slope matrix
            lower_Matrix = torch.diag(self.slopes[:,0])

            upper_Vector = self.shifts[:,1]
            lower_Vector = self.shifts[:,0]

            return self.last.back_sub_from_top_layer(steps-1, upper_Matrix, lower_Matrix, upper_Vector, lower_Vector)

        else:
            return self.bounds


    def back_sub_from_top_layer(self, steps, upper_Matrix, lower_Matrix, upper_Vector, lower_Vector):
        current_node = self

        upper_Slope_Matrix = torch.diag(self.slopes[:,1]) #diagonal upper slope matrix
        lower_Slope_Matrix = torch.diag(self.slopes[:,0])

        Upper_Boundary_Matrix = torch.matmul(upper_Matrix, upper_Slope_Matrix)
        Upper_Boundary_Vector = torch.matmul(upper_Matrix, self.shifts[:,1]) + upper_Vector
        Lower_Boundary_Matrix = torch.matmul(lower_Matrix, lower_Slope_Matrix)
        Lower_Boundary_Vector = torch.matmul(lower_Matrix, self.shifts[:,0]) + lower_Vector

        #print(f"Upper Boundary Matrix:\n{Upper_Boundary_Matrix}\n=====================================")
        #print(f"Upper Boundary Vector:\n{Upper_Boundary_Vector}\n=====================================")
        #print(f"Lower Boundary Matrix:\n{Lower_Boundary_Matrix}\n=====================================")
        #print(f"Lower Boundary Vector:\n{Lower_Boundary_Vector}\n=====================================")

        if steps > 0 and current_node.last.last.last is not None:
            return self.last.back_sub_from_top_layer(steps-1, Upper_Boundary_Matrix, Lower_Boundary_Matrix, Upper_Boundary_Vector, Lower_Boundary_Vector)

        else:
            Upper_Boundary_Pos = torch.clamp(Upper_Boundary_Matrix, min=0)
            Upper_Boundary_Neg = torch.clamp(Upper_Boundary_Matrix, max=0)
            Lower_Boundary_Pos = torch.clamp(Lower_Boundary_Matrix, min=0)
            Lower_Boundary_Neg = torch.clamp(Lower_Boundary_Matrix, max=0)

            lower = torch.matmul(Lower_Boundary_Pos, self.last.bounds[:,0]) \
                    + torch.matmul(Lower_Boundary_Neg, self.last.bounds[:,1]) + Lower_Boundary_Vector
            upper = torch.matmul(Upper_Boundary_Pos, self.last.bounds[:, 1]) \
                    + torch.matmul(Upper_Boundary_Neg, self.last.bounds[:, 0]) + Upper_Boundary_Vector

            #print(f"lower:\n{lower}\n=====================================")
            #print(f"upper:\n{upper}\n=====================================")

            return torch.stack([lower,upper],1)


class VerifyRobustness(nn.Module):
    def __init__(self, true_label, last=None, steps_backsub=0):
        super(VerifyRobustness, self).__init__()
        self.last = last
        self.true_label = true_label
        self.steps_backsub = steps_backsub

    def forward(self, bounds):
        self.bounds = bounds
        #first simple check, see if lower bound of intended label is > all other labels
        lower_true_label = bounds[self.true_label,0]
        upper_else_label = bounds[:,1]
        if sum(lower_true_label > upper_else_label)==9: #Maybe >=? (==1 for test case!)
            return True
        #if we can not verify, we backsubstitute
        elif self.steps_backsub > 0:
            #comparing if lower bound of true_label - upper bound of all other labels (pairwise) >= 0
            backsub_bounds = self.back_sub(self.steps_backsub)
            print(f"final bounds:\n{backsub_bounds}\n=====================================")
            if sum(backsub_bounds <0)==0:
                return True
            else:
                return False
        else:
            return False

    def back_sub(self, steps):
        #we interpret this as a backsubstition in an affine layer, with weight 1 for the true_label
        #and weight -1 for all other labels.
        w = torch.ones_like(self.bounds[:,0])*-1 #set weight of other labels = -1
        weights = torch.diag(w)
        weights[:,self.true_label] = 1 #set weight of this label = 1
        #remove the line with the true label
        weights = torch.cat((weights[:self.true_label],weights[self.true_label+1:]))
        bias = torch.zeros_like(self.bounds[:,0])

        return self.last.back_sub_from_top_layer(steps-1, weights, weights, bias, bias)


