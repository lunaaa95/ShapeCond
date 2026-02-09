import numpy as np
import torch


def Aug_data(sample, args):
    aug_str = args.aug
    aug_list = aug_str.split('_')
    aug_func_dic = {
        "None":identical,
        'jitter':jitter,
        'scale':scaling,
        "sl_invariant_scale": sl_invariant_scale,
        "sl_invariant_jitter": sl_invariant_jitter,
    }
    for aug in aug_list:
        sample = aug_func_dic[aug](sample, args)
    return sample

def identical(sample, args):
    return sample

def jitter(x, args):
    # https://arxiv.org/pdf/1706.00527.pdf
    # return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    return x + torch.normal(mean=0., std=args.jitter_ratio, size=x.shape).to(args.device)

def scaling(x, args):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = torch.normal(mean=1., std=args.jitter_scale_ratio, size=(x.shape[0], x.shape[2])).to(args.device)
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        product = torch.mul(xi, factor[:, :]).unsqueeze(1)
        ai.append(product)
    result = torch.cat(ai, dim=1)
    return result

def sl_invariant_jitter(x, args):
    return x * args.mask + jitter(x, args) * (1 - args.mask)
        
def sl_invariant_scale(x, args):
    return x * args.mask + scaling(x, args) * (1 - args.mask)
 
