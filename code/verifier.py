import argparse
from networks import FullyConnected
from dp_transformers_final import *

DEVICE = 'cpu'
INPUT_SIZE = 28

def analyze(net, inputs, eps, true_label):
    STEPS_BACKSUB = 30
    net.eval()

    #run box as first heuristic -> all values approximated as box
    deep_poly = DeepPolyInstance(net, eps, inputs, true_label, STEPS_BACKSUB, box=True)
    verifier_net = deep_poly.v_net
    bounds = verifier_net(inputs)
     #print(f"Bounds given back:\n{bounds}\n=====================================")

    if sum(bounds[:,0] < 0) == 0:
       return True

    # run least-area heuristic, if box was unable to verify
    deep_poly = DeepPolyInstance(net, eps, inputs, true_label, STEPS_BACKSUB, box=False, best_slope=False)
    verifier_net = deep_poly.verifier_net()
    bounds = verifier_net(inputs)

    if sum(bounds[:,0] <0) == 0:
       # print(f"Bounds given back:\n{bounds}\n=====================================")
       return True

    #run best slope analysis, if non of the previous verified
    deep_poly = DeepPolyInstance(net, eps, inputs, true_label, STEPS_BACKSUB, box=False, best_slope=True)
    return optimizeSlopes(deep_poly).optSlopes()


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
        #with open("test_results.txt", "a") as f:
        #    f.write("{},{},{}".format(args.net, args.spec, "verified\n"))
    else:
        print('not verified')
        #with open("test_results.txt", "a") as f:
        #    f.write("{},{},{}".format(args.net, args.spec, "not verified\n"))


if __name__ == '__main__':
    main()
