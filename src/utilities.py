import torch

def spu(x):
    return torch.where(x > 0, x**2 - 0.5, torch.sigmoid(-x) - 1)

def box_transformer(inputs, l, u):
    a_lower = torch.full_like(inputs, min(spu(l), spu(u)))
    a_upper = torch.full_like(inputs, max(spu(l), spu(u)))

    return a_lower, a_upper

def spu_transformer_simple(inputs, l, u):
    ### case 1: interval is non-positive
    # value range of sigmoid(-x) - 1 is [-0.5, 0] for x <= 0
    # lower line: constant at -0.5
    # upper line: constant at 0
    if u <= 0:
        a_lower = torch.full_like(inputs, -0.5)

        a_upper = torch.full_like(inputs, 0)
    
    ### case 2: interval is non-negative
    # lower line: constant at -0.5
    # upper line: use line between SPU(l) and SPU(u)
    if l >= 0:
        a_lower = torch.full_like(inputs, -0.5)
        
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

def spu_transformer_complex(inputs, l, u):
    ### case 1: interval is non-positive
    # value range of sigmoid(-x) - 1 is [-0.5, 0] for x <= 0
    # lower line: use line between SPU(l) and SPU(u)
    # upper line: constant at 0
    if u <= 0:
        slope = (spu(u) - spu(l)) / (u - l)
        intercept = torch.square(inputs) - slope*inputs - 0.5
        a_lower = torch.full_like(inputs, slope*inputs + intercept)

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