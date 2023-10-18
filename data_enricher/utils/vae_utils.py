import torch
from torch.nn.functional import softplus
from torch import inf

def to_sigma(sigmatilde):       return torch.nn.functional.softplus(sigmatilde,beta=1,threshold=5)

def from_sigma(sigma):      return sigma + torch.log(-torch.expm1(-sigma))

def log_prob(dist="Normal", params=None, targets=None):
    if dist=="Normal":
        return -0.5*torch.log(2*torch.tensor(torch.pi)) - torch.log(params["sigma"]) - 0.5*((targets-params["mu"])/params["sigma"])**2
    else:
        raise ValueError("Unknown distribution.")

def kl_divergence(dist="Normal", params=None, prior_params=None):
    if dist=="Normal":
        var_ratio = (params["sigma"] / prior_params["sigma"]).pow(2)
        t1 = ((params["mu"] - prior_params["mu"]) / prior_params["sigma"]).pow(2)
        return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
    else:
        raise ValueError("Unknown distribution.")