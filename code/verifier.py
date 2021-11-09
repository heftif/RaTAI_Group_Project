import argparse
import torch
from networks import FullyConnected

DEVICE = 'cpu'
INPUT_SIZE = 28

def spu(x):
    ''' only represents the non-negative part of the SPU! '''
    return x**2 - 0.5

def spu_transformer(inputs, l, u):
    ### case 1: interval is non-positive
    # value range of sigmoid(-x) - 1 is [-0.5, 0] for x <= 0
    # lower line: constant at -0.5
    # upper line: constant at 0
    if u <= 0:
        a_lower = torch.full_like(inputs, -0.5)
        a_upper = torch.full_like(inputs, 0)
    
    ### case 2: interval is non-negative
    # lower line: use slope SPU'(x) = 2x with intercept = x^2 - 2x - 0.5
    # upper line: use line between SPU(l) and SPU(u)
    if l >= 0:
        slope = 2
        intercept = torch.square(inputs) - slope*inputs - 0.5
        a_lower = torch.full_like(inputs, slope*inputs + intercept)
        
        slope = (spu(u) - spu(l)) / (u - l)
        intercept = torch.square(inputs) - slope*inputs - 0.5
        a_upper = torch.full_like(inputs, slope*inputs + intercept)

    ### case 3: interval crosses 0
    # lower line: constant at -0.5
    # upper line: use line between SPU(l) and SPU(u) as in case 2
    else:
        a_lower = torch.full_like(inputs, -0.5)

        slope = (spu(u) - spu(l)) / (u - l)
        intercept = torch.square(inputs) - slope*inputs - 0.5
        a_upper = torch.full_like(inputs, slope*inputs + intercept)

    return a_lower, a_upper


def analyze(net, inputs, eps, true_label):
    # denomalize

    # compute lower and upper bounds
    l = inputs - eps
    u = inputs + eps

    # run inputs through net (with backprop)
    # how to determine single layers, i.e. transformations?

    # check that proba(true_label) > proba(any other label)

    return 0


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

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
    else:
        assert False

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
