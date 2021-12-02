import torch.nn as nn
import torch
from networks import Normalization
from networks import SPU

class DeepPolyInstance():
    def __init__(self, net, eps, inputs, true_label, steps_backsub=0, box = False):
        self.net = net
        self.eps = eps
        self.inputs = inputs
        self.true_label = true_label
        self.steps_backsub = steps_backsub
        self.box = box


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
                last = SPUTransformer(last=last, steps_backsub=self.steps_backsub, box = self.box)
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
            #print(f"INPUT:\n{input}\nINITIAL BOUNDS:\n{self.bounds}\n=====================================")
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
        #print(f"BOUNDS AFTER NORMALIZING LAYER:\n{self.bounds}\n=====================================")
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

            # check if the bounds are better than the old bounds
            valid_lower = backsub_bounds[:,0] > self.bounds[:,0]
            valid_upper = backsub_bounds[:,1] < self.bounds[:,1]
            self.bounds[valid_lower, 0] = backsub_bounds[valid_lower,0]
            self.bounds[valid_upper, 1] = backsub_bounds[valid_upper,1]

        print(f"BOUNDS AFTER AFFINE LAYER:\n{self.bounds}\n=====================================")
        assert torch.all(torch.le(self.bounds[:,0], self.bounds[:,1])) # check for all lower <= upper
        return self.bounds

    def back_sub(self, steps):
        if steps > 0:
            upper_Matrix = self.weights
            lower_Matrix = self.weights

            upper_Vector = self.bias
            lower_Vector = self.bias

            backsub_bounds = self.last.back_sub_from_top_layer(steps-1, upper_Matrix, lower_Matrix, upper_Vector, lower_Vector)
            return backsub_bounds

        else:
            return self.bounds

    def back_sub_from_top_layer(self, steps, upper_Matrix, lower_Matrix, upper_Vector, lower_Vector):

        Upper_Boundary_Matrix = torch.matmul(upper_Matrix, self.weights)
        Upper_Boundary_Vector = torch.matmul(upper_Matrix, self.bias) + upper_Vector
        Lower_Boundary_Matrix = torch.matmul(lower_Matrix, self.weights)
        Lower_Boundary_Vector = torch.matmul(lower_Matrix, self.bias) + lower_Vector

        #print(f"Upper Boundary Matrix Affine:\n{Upper_Boundary_Matrix}\n=====================================")
        #print(f"Upper Boundary Vector Affine:\n{Upper_Boundary_Vector}\n=====================================")
        #print(f"Lower Boundary Matrix Affine:\n{Lower_Boundary_Matrix}\n=====================================")
        #print(f"Lower Boundary Vector Affine:\n{Lower_Boundary_Vector}\n=====================================")

        if steps > 0 and self.last.last.last is not None:
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

            #print(f"lower affine:\n{lower}\n=====================================")
            #print(f"upper affine:\n{upper}\n=====================================")

            return torch.stack([lower,upper],1)

class SPUTransformer(nn.Module):
    def __init__(self, last=None, steps_backsub=0, box=False):
        super(SPUTransformer, self).__init__()
        self.last = last
        self.steps_backsub = steps_backsub
        self.box = box


    def forward(self, bounds):
        ''' update bounds according to SPU function '''
        spu = SPU()
        #initialise shift, slopes and param
        self.slopes = torch.zeros_like(bounds)
        self.shifts = torch.zeros_like(bounds)

        #calculate the spu values of all bounds
        val_spu = spu(bounds)

        #calculate the difference
        diff = (bounds[:,1] - bounds[:,0])

        ### case 1: interval is non-positive
        neg_ind = bounds[:,1]<0
        self.neg_ind = neg_ind
        ### case 2: interval is non-negative
        pos_ind = bounds[:,0]>=0
        self.pos_ind = pos_ind
        ### case 3: crossing
        cross_ind = torch.logical_not(torch.logical_or(neg_ind, pos_ind))
        self.cross_ind = cross_ind

        # print(f"Number Negative:\n{sum(self.neg_ind)}\n=====================================")
        # print(f"Number Positive:\n{sum(self.pos_ind)}\n=====================================")
        # print(f"Number Crossing:\n{sum(self.cross_ind)}\n=====================================")

        all_slopes = torch.div(val_spu[:,1]-val_spu[:,0], diff)

        cross_ind_pos = torch.logical_and(all_slopes > 0, self.cross_ind)
        cross_ind_neg = torch.logical_and(all_slopes < 0, self.cross_ind)

        #TODO: uncomment 2 lines below to get slopes again and not only calculate with all boxes
        #calculate the upper slopes (for purely positive intervals, for the others it's 0=>constant lower bound)
        self.slopes[pos_ind,1] = all_slopes[pos_ind]
        #calculate the lower slopes (for purely negative intervals, for the others it's 0=>constant upper bound)
        self.slopes[neg_ind,0] = all_slopes[neg_ind]
        #upper bound of crossing indexes, if we don't approximate everything by a box anyway
        if not self.box:
            self.slopes[cross_ind_pos,1] = all_slopes[cross_ind_pos]


        #TODO: refine the slopes for cross_ind_neg, they should be approximated better, using the turning point of the
        #TODO: sigmoid function => where the second derivation = 0 (I think)
        #TODO: so everywhere, where val_spu[:,0] < turning point, use the upper slope: self.slopes[cross_ind_neg > tp,1] = all_slopes[cross_ind_neg > tp]

        #print(f"SLOPES in SPU:\n{self.slopes}\n=====================================")
        self.ind_switched = torch.zeros_like(bounds[:,0])
        self.ind_switched = all_slopes < 0

        #switch the val_spu
        new_upper = val_spu[self.ind_switched,0]
        new_lower = val_spu[self.ind_switched,1]
        val_spu[self.ind_switched,0] = new_lower
        val_spu[self.ind_switched,1] = new_upper

        #calculate the new bounds -> just take the function value.
        self.bounds = val_spu

        #set the lower bounds of crossing indexes to -0.5
        self.bounds[self.cross_ind,0] = -0.5

        #need to switch the input bounds as well, else we get invalid points when calculating the shifts
        new_upper_2 = bounds[self.ind_switched,0]
        new_lower_2 = bounds[self.ind_switched,1]
        bounds[self.ind_switched,0] = new_lower_2
        bounds[self.ind_switched,1] = new_upper_2

        #calculate the shifts (to get the full linear description (y=slope*x + shift))
        #if the slope is zero, the shift is just set to the constant bound value
        self.shifts[:,1] = val_spu[:,1] - self.slopes[:,1]*bounds[:,1]
        self.shifts[:,0] = val_spu[:,0] - self.slopes[:,0]*bounds[:,0]

        #set the constant shifts
        self.shifts[self.cross_ind,0] = -0.5

        #if we approximate by box,
        #if self.box:
        #    self.shifts[cross_ind_neg,1] = self.bounds[cross_ind_neg,1]


        #some test plots
        #y = np.linspace(-20,8,10000)
        #y_tensor = torch.from_numpy(y)

        #plt.plot(y_tensor,spu(y_tensor))
        #plt.axis([-10, 1, -0.5, 5])
        #plt.plot( bounds[1,0],val_spu[1,0], 'ro')
        #plt.plot(bounds[1,1],val_spu[1,1], 'ro')

        #y_u = torch.from_numpy(np.linspace(-15,5,5000))
        #y_upper = self.slopes[1,1]*y_u+self.shifts[1,1]
        #y_lower = self.slopes[1,0]*y_u+self.shifts[1,0]
        #plt.plot(y_u, y_upper, '--')
        #plt.plot(y_u, y_lower)


        # print(f"SHIFT NEW:\n{self.shifts}\n=====================================")
        # print(f"BOUNDS after SPU, before backsub:\n{self.bounds}\n=====================================")

        # use backsubstitution in case it is requested
        if self.steps_backsub > 0:
            backsub_bounds = self.back_sub(self.steps_backsub)

            # check if the bounds are better than the old bounds
            valid_lower = backsub_bounds[:,0] > self.bounds[:,0]
            valid_upper = backsub_bounds[:,1] < self.bounds[:,1]
            self.bounds[valid_lower, 0] = backsub_bounds[:,0][valid_lower]
            self.bounds[valid_upper, 1] = backsub_bounds[:,1][valid_upper]

        #print(f"BOUNDS after SPU, with backsub:\n{self.bounds}\n=====================================")
        assert torch.all(torch.le(self.bounds[:,0], self.bounds[:,1])) # check for all lower <= upper
        return self.bounds

    #for when we do the first backsubstitution
    def back_sub(self, steps):
        if steps > 0:
            upper_Matrix = torch.diag(self.slopes[:,1]) # diagonal upper slope matrix
            lower_Matrix = torch.diag(self.slopes[:,0])

            upper_Vector = self.shifts[:,1]
            lower_Vector = self.shifts[:,0]

            backsub_bounds = self.last.back_sub_from_top_layer(steps-1, upper_Matrix, lower_Matrix, upper_Vector, lower_Vector)
            return backsub_bounds

        else:
            return self.bounds


    def back_sub_from_top_layer(self, steps, upper_Matrix, lower_Matrix, upper_Vector, lower_Vector):

        upper_Slope_Matrix = torch.diag(self.slopes[:,1]) #diagonal upper slope matrix
        lower_Slope_Matrix = torch.diag(self.slopes[:,0]) #diagonal lower slope matrix

        #the same clamping that we did in the affine layer in the beginning, we have to do here
        #else we might get upper bounds > lower bounds
        #easiest to imagine if you approximate everything by box approximation => all terms with slopes fall away
        #then, the bounds are determined by: weights*(shifts)+ bias, which needs to give the same results as
        #before, when we calculated weights*(bounds)+bias => apply the same clamping here!

        #works fine for the lower boundary vector and upper boundary vector,
        #not yet when we include the slopes again, because for us, slopes can also be negative, not only >= 0 as ReLU
        #we have to take into account the sign of the slopes and THEN clamp the weight
        #but we can't use different weights to calculate the boundary_matrix and the boundary_vector, would be inconsistent
        #for upper slopes it doesn't actually matter, because they are always positive, but let's do it anyway for consistency

        #TODO: Figure out the hack in the upper and lower boundary matrix
        #TODO: could also be because we need to switch the bounds for negative values maybe?
        #maybe not, because when I uncomment the slopes, I still get an error, so the upper_boundary_matrix and the
        #lower boundary matrix must still have a hack...

        #upper_neg = upper_Slope_Matrix < 0
        #lower_neg = lower_Slope_Matrix < 0

        #upper_Matrix[upper_neg] = upper_Matrix[upper_neg]*-1
        #lower_Matrix[lower_neg] = lower_Matrix[lower_neg]*-1

        Upper_Boundary_Matrix= torch.matmul(torch.clamp(upper_Matrix, min=0.0), upper_Slope_Matrix) + \
                               torch.matmul(torch.clamp(upper_Matrix, max=0.0), lower_Slope_Matrix)

        Upper_Boundary_Vector = torch.matmul(torch.clamp(upper_Matrix, min=0.0), self.shifts[:,1]) + \
                                torch.matmul(torch.clamp(upper_Matrix, max=0.0), self.shifts[:,0]) + upper_Vector

        Lower_Boundary_Matrix = torch.matmul(torch.clamp(lower_Matrix, min=0.0), lower_Slope_Matrix) + \
                                torch.matmul(torch.clamp(lower_Matrix, max=0.0), upper_Slope_Matrix)

        Lower_Boundary_Vector = torch.matmul(torch.clamp(lower_Matrix, max=0.0), self.shifts[:,1]) + \
                                torch.matmul(torch.clamp(lower_Matrix, min=0.0), self.shifts[:,0])+ lower_Vector

        # print(f"First Row Upper Matrix:\n{upper_Matrix[0,:]}\n=====================================")
        # print(f"upper shifts:\n{self.shifts[:,1]}\n=====================================")

        if steps > 0:
            return self.last.back_sub_from_top_layer(steps-1, Upper_Boundary_Matrix, Lower_Boundary_Matrix, Upper_Boundary_Vector, Lower_Boundary_Vector)

        else:
            Upper_Boundary_Pos = torch.clamp(Upper_Boundary_Matrix, min=0)
            Upper_Boundary_Neg = torch.clamp(Upper_Boundary_Matrix, max=0)
            Lower_Boundary_Pos = torch.clamp(Lower_Boundary_Matrix, min=0)
            Lower_Boundary_Neg = torch.clamp(Lower_Boundary_Matrix, max=0)

            #maybe perform bound switch, if we substitute in here...
            #because we switch them in the spu layer to calculate the bounds that follow
            #fixed_bounds = torch.clone(self.last.bounds).detach()
            #new_lower = fixed_bounds[self.ind_switched,1]
            #new_upper = fixed_bounds[self.ind_switched,0]
            #fixed_bounds[self.ind_switched,0] = new_lower
            #fixed_bounds[self.ind_switched,1] = new_upper

            lower = torch.matmul(Lower_Boundary_Pos, self.last.bounds[:,0]) \
                    + torch.matmul(Lower_Boundary_Neg, self.last.bounds[:,1]) + Lower_Boundary_Vector
            upper = torch.matmul(Upper_Boundary_Pos, self.last.bounds[:, 1]) \
                    + torch.matmul(Upper_Boundary_Neg, self.last.bounds[:, 0]) + Upper_Boundary_Vector


            #print(f"Upper Boundary Matrix SPU:\n{Upper_Boundary_Matrix}\n=====================================")
            #print(f"Upper Boundary Bias SPU:\n{upper_Vector}\n=====================================")
            #print(f"Upper Boundary Vec Part1 SPU:\n{Up_Boundary_Vector2}\n=====================================")
            #print(f"Upper Boundary Vector SPU:\n{Upper_Boundary_Vector}\n=====================================")
            #print(f"Lower Boundary Matrix SPU:\n{Lower_Boundary_Matrix}\n=====================================")
            #print(f"Lower Boundary Vector SPU:\n{Lower_Boundary_Vector}\n=====================================")


            return torch.stack([lower,upper],1)


class VerifyRobustness(nn.Module):
    def __init__(self, true_label, last=None, steps_backsub=0):
        super(VerifyRobustness, self).__init__()
        self.last = last
        self.true_label = true_label
        self.steps_backsub = steps_backsub

    def forward(self, bounds):
        #we interpret this as a backsubstition in an affine layer, with weight 1 for the true_label
        #and weight -1 for all other labels.

        #create weights and bias
        w = torch.ones_like(bounds[:,0])*-1 #set weight of other labels = -1
        self.weights_var1 = torch.diag(w)
        self.weights_var1[:,self.true_label] = 1 #set weight of this label = 1
        #remove the line with the true label
        self.weights_var1 = torch.cat((self.weights_var1[:self.true_label],self.weights_var1[self.true_label+1:]))
        self.bias = torch.zeros_like(bounds[1:len(bounds), 0])

        #calculate the bounds (exactly the same as affine transformation)
        positive_weights_var1 = torch.clamp(self.weights_var1, min=0)
        negative_weights_var1 = torch.clamp(self.weights_var1, max=0)

        lower = torch.matmul(positive_weights_var1, bounds[:,0]) + torch.matmul(negative_weights_var1, bounds[:,1]) # bounds[:,0] are all lower bounds, bounds[:,1] are all upper bounds
        upper = torch.matmul(positive_weights_var1, bounds[:,1]) + torch.matmul(negative_weights_var1, bounds[:,0])
        self.bounds_var1 = torch.stack([lower, upper], 1)
        if self.bias is not None:
            self.bounds_var1 += self.bias.reshape(-1, 1) # add the bias where it exists
        self.bounds_var1 = torch.stack([lower,upper],1)

        #print(f"final bounds before backsub:\n{self.bounds_var1}\n=====================================")
        #print("=====================================")


        #if we can not verify, we backsubstitute
        if self.steps_backsub > 0:
            #the backsub_bounds yield the upper and lower bounds for the difference
            #if the lower values is < 0, the pairing could yield values below zero and thus is not verified
            backsub_bounds = self.back_sub(self.steps_backsub, self.weights_var1, self.bias)

            valid_lower = backsub_bounds[:,0] > self.bounds_var1[:,0]
            valid_upper = backsub_bounds[:,1] < self.bounds_var1[:,1]
            self.bounds_var1[valid_lower, 0] = backsub_bounds[:,0][valid_lower]
            self.bounds_var1[valid_upper, 1] = backsub_bounds[:,1][valid_upper]

            #print(f"final bounds after Backsub:\n{self.bounds_var1}\n=====================================")

        #then, we do exactly the same thing, but for the reversed matrix, where all other factors are 1, and the
        #weights of the true label is -1
        #not so sure about this thing here yet!

        # #create weights and bias
        # w = torch.ones_like(bounds[:,0])*1 #set weight of other labels = -1
        # self.weights_var2 = torch.diag(w)
        # self.weights_var2[:,self.true_label] = -1 #set weight of this label = 1
        # #remove the line with the true label
        # self.weights_var2 = torch.cat((self.weights_var2[:self.true_label],self.weights_var2[self.true_label+1:]))
        # self.bias = torch.zeros_like(bounds[1:len(bounds), 0])
        #
        # #calculate the bounds (exactly the same as affine transformation=
        # positive_weights_var2 = torch.clamp(self.weights_var1, min=0)
        # negative_weights_var2 = torch.clamp(self.weights_var1, max=0)
        #
        # lower = torch.matmul(positive_weights_var2, bounds[:,0]) + torch.matmul(negative_weights_var2, bounds[:,1]) # bounds[:,0] are all lower bounds, bounds[:,1] are all upper bounds
        # upper = torch.matmul(positive_weights_var2, bounds[:,1]) + torch.matmul(negative_weights_var2, bounds[:,0])
        # self.bounds_var2 = torch.stack([lower, upper], 1)
        # if self.bias is not None:
        #     self.bounds_var2 += self.bias.reshape(-1, 1) # add the bias where it exists
        # self.bounds_var2 = torch.stack([lower,upper],1)
        #
        # print(f"final bounds before backsub:\n{self.bounds_var2}\n=====================================")
        # print("=====================================")
        #
        #
        # #we backsubstitute
        # if self.steps_backsub > 0:
        #     #the backsub_bounds yield the upper and lower bounds for the difference
        #     #if the lower values is < 0, the pairing could yield values below zero and thus is not verified
        #     backsub_bounds = self.back_sub(self.steps_backsub, self.weights_var2, self.bias)
        #
        #     valid_lower = backsub_bounds[:,0] > self.bounds_var2[:,0]
        #     valid_upper = backsub_bounds[:,1] < self.bounds_var2[:,1]
        #     self.bounds_var2[valid_lower, 0] = backsub_bounds[:,0][valid_lower]
        #     self.bounds_var2[valid_upper, 1] = backsub_bounds[:,1][valid_upper]
        #
        #     print(f"final bounds after Backsub:\n{self.bounds_var2}\n=====================================")

        #return self.bounds_var1, self.bounds_var2

        return self.bounds_var1

    def back_sub(self, steps, weights, bias):
        return self.last.back_sub_from_top_layer(steps-1, weights, weights, bias, bias)


