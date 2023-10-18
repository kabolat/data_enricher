import torch
from ..utils.vae_utils import *

ACTIVATION = torch.nn.ELU()

class NNBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons=50, num_hidden_layers=2, **_):
        super(NNBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.num_layers = num_hidden_layers

        self.input_layer = torch.nn.Linear(input_dim, num_neurons)
        self.middle_layers = torch.nn.ModuleList([torch.nn.Linear(num_neurons, num_neurons) for _ in range(num_hidden_layers)]) 
        self.output_layer = torch.nn.Linear(num_neurons, output_dim)

        # setup the non-linearity
        self.act = ACTIVATION

    def forward(self, x):
        h = x.view(-1, self.input_dim)
        h = self.act(self.input_layer(h))
        
        for layer in self.middle_layers:
            h = self.act(layer(h))

        h = self.output_layer(h)
        return h

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ParameterizerNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dist_params=["mu"], num_hidden_layers=2, num_neurons=50, **_):
        super(ParameterizerNN, self).__init__()
        self.dist_params = dist_params
        self.block_dict = torch.nn.ModuleDict()

        self.block_dict["input"] = NNBlock(input_dim, num_neurons, num_neurons=num_neurons, num_hidden_layers=num_hidden_layers)

        for param in dist_params:
            self.block_dict[param] = NNBlock(num_neurons, output_dim, num_neurons=num_neurons, num_hidden_layers=1)
        # setup the non-linearity
        self.act = ACTIVATION

    def forward(self, inputs):
        h = inputs.view(-1, self.block_dict["input"].input_dim)
        h = self.act(self.block_dict["input"](h))
        output_dict = {}
        for param in self.dist_params:
            output_dict[param] = self.block_dict[param](h)
        return output_dict
    
    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GaussianNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, learn_sigma=True, num_hidden_layers=2, num_neurons=50, **_):
        super(GaussianNN, self).__init__()

        self.learn_sigma = learn_sigma

        if learn_sigma: dist_params = ["mu", "sigma"]
        else: dist_params = ["mu"]

        self.parameterizer = ParameterizerNN(input_dim, output_dim, dist_params=dist_params, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons)

    def _num_parameters(self):
        return self.parameterizer._num_parameters()
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        if self.learn_sigma: param_dict["sigma"] = to_sigma(param_dict["sigma"])
        else: param_dict["sigma"] = torch.ones_like(param_dict["mu"])
        return param_dict
    
    def rsample(self, param_dict=None, num_samples=1, **_):
        return param_dict["mu"] + param_dict["sigma"] * torch.randn((num_samples, *param_dict["mu"].shape))

    def sample(self, param_dict=None, num_samples=1, **_):
        return self.rsample(param_dict=param_dict, num_samples=num_samples)
    
    def log_likelihood(self, targets, param_dict=None):
        return log_prob("Normal", param_dict, targets)
    
    def kl_divergence(self, param_dict=None, prior_params={"mu":0.0, "sigma":1.0}):
        return kl_divergence("Normal", param_dict, prior_params)

def get_distribution_model(dist_type, **kwargs):
    if dist_type.lower() in ["gaussian", "gauss", "normal", "n", "g"]: return GaussianNN(**kwargs)
    else: raise NotImplementedError("Unknown distribution type: {}".format(dist_type))

def get_prior_params(dist_type, num_dims=1):
    if dist_type.lower() in ["gaussian", "gauss", "normal", "n", "g"]: return {"mu":torch.zeros(num_dims), "sigma":torch.ones(num_dims)}
    else: raise NotImplementedError("Unknown distribution type: {}".format(dist_type))