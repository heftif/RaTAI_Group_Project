import argparse
import torch
import torchvision
import numpy as np
from networks import FullyConnected
from networks import Normalization
from networks import SPU
from utilities import *
from dp_transformers import *
from example_network import *

DEVICE = 'cpu'
INPUT_SIZE = 28

class dummyLayers:
    def __init__(self):
        self.type = []
        self.lowerBounds = []
        self.upperBounds = []
        self.operations = []

class operations:
    def __init__(self, layer, inputFormat, outputFormat, mean=0, std=0, weights=[], bias=[]):
        self.layer= layer
        self.inputFormat = inputFormat
        self.outputFormat = outputFormat
        self.mean = mean
        self.std = std
        self.weights = weights
        self.bias = bias

def normalize(x):
    mean = 0.1307
    sigma = 0.3081
    return (x-mean)/sigma


def analyze_test(net, inputs, eps, true_label):
    pixels = 784

    #normalize input the data (here? or after -eps/+eps? or both?
    inputs = normalize(inputs)

    # compute lower and upper bounds
    #inputs-eps does not set values smaller than zero -> is that correct? Should that be the case?
    l = inputs - eps
    u = inputs + eps

    #do we need to normalize the input again?

    #cut lower and upper bounds at 0 and 1, as the task states we have to verify only for possible input
    l[l<0] = 0
    u[u>1] = 1

    #extract weights and biases from linear layers
    l = net.state_dict()

    #initialise object arrays
    operation = []
    #read the network
    #get all the layers with their corresponding in and out features
    for i in range(0, len(net.layers)):
        layer = net.layers[i]
        if isinstance(layer, Normalization):
            operation.append(operations('Normalization', 784, 784, layer.mean.item(), layer.sigma.item()))
        elif isinstance(layer, SPU):
            operation.append(operations('SPU', operation[i-2].outputFormat, operation[i-2].outputFormat))
        elif isinstance(layer, torch.nn.Linear):
            #extracts the weights and bias of the layers
            operation.append(operations('Linear', layer.in_features, layer.out_features,
                                        weights =l['layers.'+ str(i) + '.weight'], bias = l['layers.'+ str(i) + '.bias']))
        #print(layer)
        #what about the flatten layer?
        #maybe think about setting the predecessor


    #create a dummy network object, where we can push the data through
    dummyNetwork = dummyLayers()
    dummyNetwork.upperBounds = u
    dummyNetwork.lowerBounds = l
    dummyNetwork.operations = operation

    #ERAN Code: look at line 1400 in main -> go to eran analyze box -> line 95 deep poly -> follow the functions
    #for the actual analization of the given network, similar to what I set up until here, go to analyzer.analyze
    #in eran -> get_abstract0 -> transformer -> deeppoly_nodes.py -> calc_bounds

    #see deeppoly_nodes.py -> calc_bounds for the next steps taht need to be implemented.

    #labels = np.linspace(0,9,num =10)
    #compare all labels with all adversarial labels
    #for label in labels:
        #for adv_label in labels:


    # check that proba(true_label) > proba(any other label)

    return 0

def analyze(net, inputs, eps, true_label):
    STEPS_BACKSUB = 3
    net.eval()
    deep_poly = DeepPolyInstance(net, eps, inputs, true_label, STEPS_BACKSUB)
    verifier_net = deep_poly.verifier_net()
    results = verifier_net(inputs)
    print(results)

    return 0


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args() # uncomment when everything ready and comment next line, which is only to test certan nets and examples
    #args = parser.parse_args('--net net0_fc2 --spec /home/angelos/Desktop/das_projects/reliable_interpr_ai/team-17-rai2021/test_cases/net0_fc2/example_img0_0.09500.txt'.split())

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net.endswith('fc1'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net.endswith('test'):
        net = ExampleNetwork(DEVICE)
        inputs = torch.tensor([[0.5], [0.2]])
        eps = 0.02
        true_label = None
        analyze(net, inputs, eps, true_label)
        return 0
    else:
        assert False

    #net.load_state_dict(torch.load('/home/angelos/Desktop/das_projects/reliable_interpr_ai/team-17-rai2021/mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE))) # change path back to ../mnist_nets etc.

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label


    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
