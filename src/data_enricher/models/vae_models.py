import torch
from .submodels import *
from ..utils.vae_utils import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
torch.autograd.set_detect_anomaly(True)

class VAE(torch.nn.Module):
    def __init__(self, 
                input_dim=None, 
                cond_dim=0,
                latent_dim=3,
                num_neurons=20,
                num_hidden_layers=2,
                learn_decoder_sigma=True,
                ):
        super(VAE, self).__init__()
        kwargs = locals()
        for key in ["self","__class__","input_dim","cond_dim"]: kwargs.pop(key)

        self.kwargs = kwargs
        for key in kwargs:
            setattr(self,key,kwargs[key])

        self.input_dim = input_dim
        self.cond_dim = cond_dim

        self.encoder = get_distribution_model("normal", input_dim=self.input_dim+self.cond_dim, output_dim=self.latent_dim, learn_sigma=True, **kwargs)
        self.decoder = get_distribution_model("normal", input_dim=self.latent_dim+self.cond_dim, output_dim=self.input_dim, learn_sigma=self.learn_decoder_sigma, **kwargs)
        self.prior_params = get_prior_params("normal", self.latent_dim)

        self.num_parameters = self.encoder._num_parameters()+self.decoder._num_parameters()
    
    def forward(self, inputs, conditions=None, mc_samples=1):
        if self.cond_dim==0: conditions = torch.Tensor(inputs.shape[0], 0)
        elif conditions==None: raise ValueError("Conditions must be provided.")

        posterior_params_dict = self.encoder(torch.cat((inputs,conditions),dim=1))
        z = self.encoder.rsample(posterior_params_dict, num_samples=mc_samples, **self.kwargs)

        likelihood_params_dict = self.decoder(torch.cat((z,conditions.unsqueeze(0).repeat_interleave(mc_samples,dim=0)),dim=2))
        for param in likelihood_params_dict:
            likelihood_params_dict[param] = likelihood_params_dict[param].view(mc_samples,inputs.shape[0],self.input_dim)
        
        return {"params":likelihood_params_dict}, {"params":posterior_params_dict, "samples": z}

    def sample(self, num_samples_prior=1, num_samples_likelihood=1, conditions=None):
        if self.cond_dim==0: conditions = torch.Tensor([])
        elif conditions==None: raise ValueError("Conditions must be provided.")
        conditions = conditions.unsqueeze(0).unsqueeze(0).repeat_interleave(num_samples_prior,dim=0)

        with torch.no_grad():
            z = self.encoder.sample(param_dict=self.prior_params, num_samples=num_samples_prior).unsqueeze(1)
            param_dict = self.decoder(torch.cat((z,conditions),dim=2))
            samples = self.decoder.sample(param_dict, num_samples=num_samples_likelihood)
            return {"params":param_dict, "samples": samples}
    
    def impute(self, x, conditions=None, num_steps=100, use_mean=False):
        miss_idx = x.isnan()
        x[miss_idx] = 0.0
        for _ in range(num_steps):
            x_dict, z_dict = self.reconstruct(x, conditions=conditions, mc_samples=1)
            if use_mean: x[miss_idx] = x_dict["params"]["mu"].squeeze()[miss_idx]
            else: x[miss_idx] = x_dict["samples"].squeeze()[miss_idx]
        return x
    
    def reconstruct(self, inputs, conditions=None, mc_samples=1):
        with torch.no_grad():
            x_dict, z_dict = self.forward(inputs, conditions, mc_samples)
            x_dict["samples"] = self.decoder.sample(x_dict["params"], num_samples=1)
            return x_dict, z_dict
    
    def reconstruction_loglikelihood(self, x, likelihood_params):
        return self.decoder.log_likelihood(x, likelihood_params).sum(dim=2).mean(dim=0)
    
    def kl_divergence(self, posterior_params, prior_params=None):
        if prior_params is None: prior_params = self.prior_params
        return self.encoder.kl_divergence(posterior_params, prior_params=self.prior_params).sum(dim=1)
    
    def loss(self, x, likelihood_params, posterior_params, prior_params=None, beta=1.0):
        rll = self.reconstruction_loglikelihood(x, likelihood_params).mean(dim=0)
        kl = self.kl_divergence(posterior_params,prior_params=prior_params).mean(dim=0)
        loss = -(rll-beta*kl)
        return {"loss":loss, "elbo": rll-kl, "rll": rll, "kl": kl}
    
    def train(self, 
            x,
            conditions=None,
            beta=1.0,
            mc_samples=1,
            learning_rate=1e-3,
            epochs=500,
            verbose_freq=100,
            batch_size=32,):
        
        if self.cond_dim==0: conditions = torch.Tensor(x.shape[0], 0)
        elif conditions==None: raise ValueError("Conditions must be provided.")
        dataset = torch.utils.data.TensorDataset(x, conditions)

        optim = torch.optim.Adam([{'params':self.encoder.parameters()}, {'params':self.decoder.parameters()}], lr=learning_rate)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        total_itx = ((dataset.__len__())//batch_size + (dataloader.drop_last==False))*epochs

        epx, itx = 0, 0
        for _ in range(epochs):
            epx += 1

            for inputs, conditions in dataloader:
                itx += 1

                # region Forward and Backward Pass
                x_dict, z_dict = self.forward(inputs, conditions, mc_samples=mc_samples)
                optim.zero_grad()
                loss = self.loss(inputs, x_dict["params"], z_dict["params"], prior_params=self.prior_params, beta=beta)
                loss["loss"].backward()
                optim.step()
                #endregion
                
                # region Logging
                if itx%verbose_freq==0: 
                    print(f"Iteration: {itx}/{total_itx} -- ELBO={loss['elbo'].item():.2e} / RLL={loss['rll'].item():.2e} / KL={loss['kl'].item():.2e}")
                #endregion