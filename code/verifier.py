import argparse
import torch
import numpy as np
from networks import FullyConnected
from utilities import *
from dp_transformers import *
from example_network import *

DEVICE = 'cpu'
INPUT_SIZE = 28

def analyze(net, inputs, eps, true_label):
    STEPS_BACKSUB = 20
    net.eval()

    # run box as first heuristic -> all values approximated as box
    deep_poly = DeepPolyInstance(net, eps, inputs, true_label, STEPS_BACKSUB, box=True)
    verifier_net = deep_poly.verifier_net()
    bounds = verifier_net(inputs)
    # print(f"Bounds given back:\n{bounds}\n=====================================")

    if sum(bounds[:,0] < 0) == 0:
        return True

    # run more sophisticated heuristics if box was unable to verify
    deep_poly = DeepPolyInstance(net, eps, inputs, true_label, STEPS_BACKSUB, box=False)
    verifier_net = deep_poly.verifier_net()
    bounds = verifier_net(inputs)
    # print(f"Bounds given back:\n{bounds}\n=====================================")

    if sum(bounds[:,0] <0) == 0:
        return True

    # in case all heuristics fail to verify
    return False


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')

    args = parser.parse_args() # uncomment when everything ready and comment next line, which is only to test certan nets and examples
    # args = parser.parse_args('--net net1_fc5 --spec /home/angelos/Desktop/das_projects/reliable_interpr_ai/team-17-rai2021/prelim_test_cases/net1_fc5/final_img6_0.03500.txt'.split())

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
        true_label = 0
        ver = analyze(net, inputs, eps, true_label)
        if(ver):
            print("verified")
        else:
            print("not verified")
        return 0
    elif args.net.endswith('test2'):
        net = ExampleNetwork_nocrossing(DEVICE)
        inputs = torch.tensor([[0.5], [0.2]])
        eps = 0.02
        true_label = 0
        ver = analyze(net, inputs, eps, true_label)
        if(ver):
            print("verified")
        else:
            print("not verified")
        return 0
    else:
        assert False

    # net.load_state_dict(torch.load('/home/angelos/Desktop/das_projects/reliable_interpr_ai/team-17-rai2021/mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE))) # change path back to ../mnist_nets etc.
    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

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
