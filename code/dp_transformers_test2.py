from numpy.core.shape_base import block
import torch.nn as nn
import torch
from networks import Normalization
from networks import SPU, derivative_spu
import numpy as np

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

        # print(f"BOUNDS AFTER AFFINE LAYER, before Backsub:\n{self.bounds}\n=====================================")

        if (self.steps_backsub > 0) and self.last.last.last is not None: # no backsub needed for the first affine layer
            backsub_bounds = self.back_sub(self.steps_backsub)

            # check if the bounds are better than the old bounds
            valid_lower = backsub_bounds[:,0] > self.bounds[:,0]
            valid_upper = backsub_bounds[:,1] < self.bounds[:,1]
            self.bounds[valid_lower, 0] = backsub_bounds[valid_lower,0]
            self.bounds[valid_upper, 1] = backsub_bounds[valid_upper,1]

        # print(f"BOUNDS AFTER AFFINE LAYER:\n{self.bounds}\n=====================================")
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
        der_spu = derivative_spu()

        #initialise shift, slopes and param
        self.slopes = torch.zeros_like(bounds)
        self.shifts = torch.zeros_like(bounds)

        #calculate the spu values of all bounds
        val_spu = spu(bounds)

        ### case 1: interval is non-positive
        neg_ind = bounds[:,1]<0
        self.neg_ind = neg_ind
        ### case 2: interval is non-negative
        pos_ind = bounds[:,0]>=0
        self.pos_ind = pos_ind
        ### case 3: crossing
        cross_ind = torch.logical_not(torch.logical_or(neg_ind, pos_ind))
        self.cross_ind = cross_ind

        # calculate all tangent slopes of the upper and lower bounds (tangent at the respective points)
        tangent_slopes = der_spu(bounds)

        #calculate the difference
        diff = (bounds[:,1] - bounds[:,0])

        #calculate all the slopes
        all_slopes = torch.div(val_spu[:,1]-val_spu[:,0], diff)

        self.shifts = torch.zeros_like(bounds)


        #print(f"Number Negative:\n{sum(self.neg_ind)}\n=====================================")
        #print(f"Number Positive:\n{sum(self.pos_ind)}\n=====================================")
        #print(f"Number Crossing:\n{sum(self.cross_ind)}\n=====================================")



        if not self.box:

            self.cross_ind_pos = torch.logical_and(all_slopes > 0, self.cross_ind)
            self.cross_ind_neg = torch.logical_and(all_slopes < 0, self.cross_ind)

            #check if negative crossing places have a slope < tangent (slope steeper than tangent) we cross the spu line
            cross_ind_neg_crossing_spu = torch.logical_and(all_slopes < tangent_slopes[:,0], self.cross_ind_neg)
            cross_ind_neg_not_crossing_spu = torch.logical_and(all_slopes >= tangent_slopes[:,0], self.cross_ind_neg)

            ## set all the fixed bounds that are not dependent on area calculations
            # calculate the upper slopes (for purely positive intervals)
            self.slopes[pos_ind,1] = all_slopes[self.pos_ind]

            #calculate the lower slopes (for purely negative intervals)
            #I set the upper bound here of the negative cases, even though it should be the lower bound
            #I will switch all the calculations for the negative_indexes in the very end, so we can
            #keep the same calculation logic, it's way easier then implementing something separate in every calculation!
            self.slopes[neg_ind,1] = all_slopes[self.neg_ind]

            #set the upper slope of positive crossing = slope
            self.slopes[self.cross_ind_pos,1]   = all_slopes[self.cross_ind_pos]

            #set the upper slope of neg crossing not crossing spu to slope
            self.slopes[cross_ind_neg_not_crossing_spu,1] = all_slopes[cross_ind_neg_not_crossing_spu]

            #set the upper slope of neg crossing, crossing spu to tangent of lower bound
            self.slopes[cross_ind_neg_crossing_spu,1] = tangent_slopes[cross_ind_neg_crossing_spu,0]


            #calculating the shifts of all upper bounds
            #we take the lower val_spu and lower bounds as reference (doesn't matter for slopes, as lower and upper points are on the line)
            #but for the tangents of cross_ind_neg_crossing_spu it matters and MUST be the lower
            self.shifts[:,1] = val_spu[:,0] - self.slopes[:,1]*bounds[:,0]

            #Calculate the new upper bound values of the crossing - crossing values - the intersection
            # between tangent line and parabola
            # spu: y = x^2 - 0.5
            # tangent line: y = upper_slope*x + shift
            # set the equations equal: x^2 - upper_slope*x - (shift+0.5) = 0
            # quadratic equation: x = (-b +- sqrt(b^2-4ac))/2a
            bounds[cross_ind_neg_crossing_spu,1] = torch.div((self.slopes[cross_ind_neg_crossing_spu,1] +
                                            torch.sqrt(torch.square(self.slopes[cross_ind_neg_crossing_spu,1]) +
                                            4*(self.shifts[cross_ind_neg_crossing_spu,1]+0.5))),2)

            #recalculate the corresponding spu value
            val_spu[cross_ind_neg_crossing_spu,1] = spu(bounds[cross_ind_neg_crossing_spu,1])

            #recalculate the corresponding upper tangent value
            tangent_slopes[cross_ind_neg_crossing_spu,1] = der_spu(bounds[cross_ind_neg_crossing_spu,1])


            #for the crossing values, we don't want the tangent as a lower bound. Instead, we want the slope
            #between [lower, -0.5] as the given bound OR a straight line at -0.5
            #for the purely positive and purely negative, we want the corresponding tangent lines
            prov_lower_slopes = torch.clone(tangent_slopes)
            prov_lower_slopes[self.cross_ind, 0] = torch.div(-0.5 - val_spu[self.cross_ind,0], 0-bounds[self.cross_ind,0])

            #torch.div(val_spu[:,1]-val_spu[:,0], diff)
            #get full linear discription of all tangent lines
            shifts_prov_lower_slopes = val_spu - prov_lower_slopes*bounds

            #constant lower shifts for horizontal lower bound approach
            constant_lower_shifts = torch.zeros_like(bounds[:,0])
            #for crossing, always equal to -0.5
            constant_lower_shifts[self.cross_ind] = -0.5
            #for positive, equal to lower bound
            constant_lower_shifts[self.pos_ind] = val_spu[self.pos_ind,0]
            #for negative, equal to lower bound
            constant_lower_shifts[self.neg_ind] = val_spu[self.neg_ind,0]

            y = torch.zeros_like(bounds)
            #evaluate the lower bound tangent on the upper bound value
            y[:,1] = prov_lower_slopes[:,0] * bounds[:,1]+shifts_prov_lower_slopes[:,0]
            #evaluate the upper bound tangent on the lower bound value
            y[:,0] = prov_lower_slopes[:,1] * bounds[:,0]+shifts_prov_lower_slopes[:,1]

            #for horizontal lower bounds to find the intersecting value with upper bound
            #y = slope*x + shift => x = (y-shift)/slope
            x = torch.div(constant_lower_shifts-self.shifts[:,1], self.slopes[:,1])


            #Calculate the triangle areas (interpreted as 1/2 of a parallelogram)
            areas = torch.zeros([bounds.size(dim=0),3])

            #area with lower bound from lower tangent
            areas[:,0] = 0.5*(torch.abs(y[:,1]-val_spu[:,1])*(bounds[:,1]-bounds[:,0]))
            #area with lower bound from upper tangent
            areas[:,1] = 0.5*(torch.abs(y[:,0]-val_spu[:,0])*(bounds[:,1]-bounds[:,0]))

            #can also be calculated for non-crossing, as it should always be > then the other triangles
            #area with constant lower bound, orthogonal triangle
            areas[x >= bounds[:,1]-1e-5,2] = 0.5*(abs(val_spu[x >= bounds[:,1]-1e-5,0]-constant_lower_shifts[x >= bounds[:,1]-1e-5]))*\
                                             (x[x >= bounds[:,1]-1e-5]-bounds[x >= bounds[:,1]-1e-5,0])
            areas[x <= bounds[:,0]+1e-5,2] = 0.5*(abs(val_spu[x <= bounds[:,0]+1e-5,1]-constant_lower_shifts[x <= bounds[:,0]+1e-5]))*\
                                             (bounds[x <= bounds[:,0]+1e-5,1] - x[x <= bounds[:,0]+1e-5])

            min_area_index = torch.argmin(areas,dim=1)
            #where option 1 has the lowest area: set lower tangent to slope
            self.slopes[min_area_index==0,0] = prov_lower_slopes[min_area_index==0,0]
            self.shifts[min_area_index==0,0] = shifts_prov_lower_slopes[min_area_index==0,0]
            #where option 2 has the lowest area: set upper tangent to slope
            self.slopes[min_area_index==1,0] = prov_lower_slopes[min_area_index==1,1]
            self.shifts[min_area_index==1,0] = shifts_prov_lower_slopes[min_area_index==1,1]
            #where option 3 has the lowest area: set constant lower bound
            self.slopes[min_area_index==2,0] = 0
            self.shifts[min_area_index==2,0] = constant_lower_shifts[min_area_index==2]

            #invert the shifts and slopes for negative cases
            new_lower_slopes_neg = torch.clone(self.slopes[self.neg_ind,1])
            new_upper_slopes_neg = torch.clone(self.slopes[self.neg_ind,0])
            new_lower_shifts_neg = torch.clone(self.shifts[self.neg_ind,1])
            new_upper_shifts_neg = torch.clone(self.shifts[self.neg_ind,0])
            self.slopes[self.neg_ind,0] = torch.clone(new_lower_slopes_neg)
            self.slopes[self.neg_ind,1] = torch.clone(new_upper_slopes_neg)
            self.shifts[self.neg_ind,0] = torch.clone(new_lower_shifts_neg)
            self.shifts[self.neg_ind,1] = torch.clone(new_upper_shifts_neg)


        #print(f"SLOPES in SPU:\n{self.slopes}\n=====================================")
        self.ind_switched = torch.zeros_like(bounds[:,0])
        self.ind_switched = all_slopes < 0

        #switch the val_spu
        new_upper = val_spu[self.ind_switched,0]
        new_lower = val_spu[self.ind_switched,1]
        val_spu[self.ind_switched,0] = torch.clone(new_lower)
        val_spu[self.ind_switched,1] = torch.clone(new_upper)

        # print(val_spu)
        #calculate the new bounds -> just take the function value.
        self.bounds = torch.clone(val_spu) ### VERY IMPORTANT: AVOID PURE ASSIGNMENT WITH "=" BECAUSE THEN THE VARIABLES BECOME RELATED TO ONE ANOTHER

        #TODO: discuss if this is really the case or not!
        #set the lower bounds of crossing indexes to -0.5
        self.bounds[self.cross_ind,0] = -0.5

        #need to switch the input bounds as well, else we get invalid points when calculating the shifts
        new_upper_2 = bounds[self.ind_switched,0]
        new_lower_2 = bounds[self.ind_switched,1]
        bounds[self.ind_switched,0] = torch.clone(new_lower_2)
        bounds[self.ind_switched,1] = torch.clone(new_upper_2)

        #set the constant shifts
        if self.box:
            #calculate the shifts (to get the full linear description (y=slope*x + shift))
            #if the slope is zero, the shift is just set to the constant bound value
            self.shifts[:,1] = val_spu[:,1] - self.slopes[:,1]*bounds[:,1]
            self.shifts[:,0] = val_spu[:,0] - self.slopes[:,0]*bounds[:,0]
            self.shifts[self.cross_ind,0] = -0.5


        # COMMENT THIS OUT BEFORE THE FINAL HAND-IN AS IT WILL CRASH THE CODE IF ASSERT FAILS
        for i in range(bounds.shape[0]):
            if i == 10:
              a=0
            #print(i)
            xx = torch.linspace(bounds[i,0],bounds[i,1], steps=1000)
            spu_xx = spu(xx)
            y_upper = self.slopes[i,1]*xx + self.shifts[i,1]
            y_lower = self.slopes[i,0]*xx + self.shifts[i,0]

            assert torch.all(y_upper >= spu_xx - 1e-4).item()
            assert torch.all(y_lower <= spu_xx + 1e-4).item()

        # # some test plots
        #import numpy as np
        #import matplotlib.pyplot as plt
        #import matplotlib
        #matplotlib.use("TkAgg")
        #torch.set_printoptions(precision=6)
        #y = np.linspace(-50,50,10000)
        #y_tensor = torch.from_numpy(y)
        # # # # # #
        #for i in range(17,bounds.shape[0]):
        # #
        # # # # #
        #    plt.figure()
        #    plt.title("node:" + str(i) + " lower: (" + "{0:.2e}".format(bounds[i,0].item()) + "," +
        #                            "{0:.2e}".format(val_spu[i,0].item()) + ") upper: (" + "{0:.2e}".format(bounds[i,1].item()) + "," +
        #                      "{0:.2e}".format(val_spu[i,1].item()) + ")")
        #    plt.plot(y_tensor,spu(y_tensor))
        #    plt.axis([-20, 20, -0.5, 370])
        #    plt.plot(bounds[i,0],val_spu[i,0], 'go')
        #    plt.plot(bounds[i,1],val_spu[i,1], 'ro')
        # #     # # # # #
        #    y_u = torch.from_numpy(np.linspace(-50,50,5000))
        #    y_upper = self.slopes[i,1]*y_u+self.shifts[i,1]
        #    y_lower = self.slopes[i,0]*y_u+self.shifts[i,0]
        #    y_lower_test = all_slopes[i]*y_u-0.4965
        #    plt.plot(y_u, y_upper, '--')
        #    plt.plot(y_u, y_lower)
        #    plt.show()


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

        # if not torch.all(torch.le(self.bounds[:,0], self.bounds[:,1])):
        # print(f"BOUNDS after SPU, with backsub:\n{self.bounds}\n=====================================")
        assert torch.all(torch.le(self.bounds[:,0], self.bounds[:,1] + torch.ones_like(self.bounds[:,1])*1e-20)) # check for all lower <= upper
        # plt.show()
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
        self.weights = torch.diag(w)
        self.weights[:, self.true_label] = 1 #set weight of this label = 1
        #remove the line with the true label
        self.weights = torch.cat((self.weights[:self.true_label], self.weights[self.true_label + 1:]))
        self.bias = torch.zeros_like(bounds[1:len(bounds), 0])

        #calculate the bounds (exactly the same as affine transformation)
        positive_weights = torch.clamp(self.weights, min=0)
        negative_weights = torch.clamp(self.weights, max=0)

        lower = torch.matmul(positive_weights, bounds[:,0]) + torch.matmul(negative_weights, bounds[:,1]) # bounds[:,0] are all lower bounds, bounds[:,1] are all upper bounds
        upper = torch.matmul(positive_weights, bounds[:,1]) + torch.matmul(negative_weights, bounds[:,0])
        self.bounds = torch.stack([lower, upper], 1)
        if self.bias is not None:
            self.bounds += self.bias.reshape(-1, 1) # add the bias where it exists
        self.bounds = torch.stack([lower, upper], 1)

        #print(f"final bounds before backsub:\n{self.bounds_var1}\n=====================================")
        #print("=====================================")


        #if we can not verify, we backsubstitute
        if self.steps_backsub > 0:
            #the backsub_bounds yield the upper and lower bounds for the difference
            #if the lower values is < 0, the pairing could yield values below zero and thus is not verified
            backsub_bounds = self.back_sub(self.steps_backsub, self.weights, self.bias)

            valid_lower = backsub_bounds[:,0] > self.bounds[:, 0]
            valid_upper = backsub_bounds[:,1] < self.bounds[:, 1]
            self.bounds[valid_lower, 0] = backsub_bounds[:, 0][valid_lower]
            self.bounds[valid_upper, 1] = backsub_bounds[:, 1][valid_upper]

            #print(f"final bounds after Backsub:\n{self.bounds_var1}\n=====================================")

        return self.bounds

    def back_sub(self, steps, weights, bias):
        return self.last.back_sub_from_top_layer(steps-1, weights, weights, bias, bias)


