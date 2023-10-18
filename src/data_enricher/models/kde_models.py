import torch
from ..utils.kde_utils import *

class VanillaKernelDensityEstimator(torch.nn.Module):
    """Vanilla Kernel Density Estimator"""
    def __init__(self, mu, sigma=None):
        """Initialize the model
        Args:
            mu (torch.Tensor): mu of the kernels
            sigma (torch.Tensor): sigma of the kernels"""
        super(VanillaKernelDensityEstimator,self).__init__()
        self.num_kernels, self.num_dims = mu.shape
        self.mu = mu.unsqueeze(0)
        if sigma is None: sigma = mu.std(0).mean()/self.num_kernels
        self.sigmatilde = from_sigma(torch.ones(1,self.num_kernels,1)*sigma)
        self.pitilde = from_pi(torch.ones(1,self.num_kernels)/self.num_kernels)

    def log_likelihood(self, x):
        """Compute the log likelihood of the model
        Args:
            x (torch.Tensor): input data
        Returns:
            log_likelihood (torch.Tensor): log likelihood of the model"""
        kernel_matrix_exponent = gaussian_kernel_exponent(x, self.mu, to_sigma(self.sigmatilde)) + torch.log(to_pi(self.pitilde))
        log_likelihood = torch.logsumexp(kernel_matrix_exponent,1)
        return log_likelihood
    
    def sample(self, num_samples=1):
        """Sample from the model
        Args:
            num_samples (int): number of samples to draw
        Returns:
            samples (torch.Tensor): samples from the model"""
        kernel_idx = torch.multinomial(to_pi(self.pitilde)[0], num_samples, replacement=True)
        mu_selected, sigma_selected = self.mu[0,kernel_idx,:], to_sigma(self.sigmatilde[0,kernel_idx])
        samples = sigma_selected * torch.randn(num_samples, self.num_dims) + mu_selected
        return samples
    
    def expectation_step(self, x, leave_one_out=False):
        """Expectation step of the EM algorithm
        Args:
            x (torch.Tensor): input data
            leave_one_out (bool): whether to leave one out
        Returns:
            responsibility_matrix (torch.Tensor): responsibility matrix"""
        kernel_matrix = torch.exp(gaussian_kernel_exponent(x, self.mu, to_sigma(self.sigmatilde)) + torch.log(to_pi(self.pitilde))) + 1e-32
        if leave_one_out: kernel_matrix = kernel_matrix*(1.0-torch.eye(self.num_kernels))
        responsibility_matrix = kernel_matrix/kernel_matrix.sum(1,keepdims=True)
        return responsibility_matrix
    
    def impute(self, x, num_steps=100):
        """Impute missing values
        Args:
            x (torch.Tensor): input data
            num_steps (int): number of steps to take
        Returns:
            x (torch.Tensor): imputed data"""
        miss_idx = x.isnan()
        x[miss_idx] = 0.0
        for _ in range(num_steps):
            kernel_idx = self.expectation_step(x, leave_one_out=False).argmax(1)
            sample = to_sigma(self.sigmatilde[0,kernel_idx]) * torch.randn(x.shape[0], self.num_dims) + self.mu[0,kernel_idx,:]
            x[miss_idx] = sample[miss_idx]
        return x
    
    def get_params(self):
        """Get the parameters of the model
        Returns:
            params (dict): parameters of the model"""
        return {'mu':self.mu, 'sigma':to_sigma(self.sigmatilde), 'pi':to_pi(self.pitilde)}
    
class AdaptiveKernelDensityEstimator(VanillaKernelDensityEstimator):
    def __init__(self, mu, sigma=0.1):
        super(AdaptiveKernelDensityEstimator,self).__init__(mu, sigma)
    
    def forward(self, x, leave_one_out=True, batch_idx=None):
        kernel_matrix_exponent = gaussian_kernel_exponent(x, self.mu, to_sigma(self.sigmatilde)) + torch.log(to_pi(self.pitilde))
        if leave_one_out:
            mask = 1.0-torch.eye(self.num_kernels)
            if batch_idx is not None: mask = mask[batch_idx]
            kernel_matrix_exponent = kernel_matrix_exponent*mask
            kernel_matrix_exponent[mask==0] = -1e32
        log_likelihood = torch.logsumexp(kernel_matrix_exponent,1)
        return log_likelihood
    
    def maximization_step(self, x, responsibility_matrix):
        x_mu = x.unsqueeze(1)-self.mu
        sigma = torch.sqrt(torch.sum(responsibility_matrix*(x_mu**2).mean(-1),0)/(responsibility_matrix.sum(0))).unsqueeze(-1)
        sigmatilde = from_sigma(sigma).unsqueeze(0)
        return sigmatilde, self.pitilde
    
    def _diff_parameterize(self):
        self.sigmatilde = torch.nn.Parameter(self.sigmatilde)

    def train(self, 
            x,
            leave_one_out=True,
            modified_em = False,
            learning_rate = 1e-2,
            bacth_size = 32,
            wait_convergence=True, 
            num_iterations=50, 
            objective_threshold=1e-4,
            verbose=False,
            verbose_freq=10):
        
        train_flag, itx, last_objective = True, 0, 0.0

        if not modified_em: 
            self._diff_parameterize()
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            dataloader = torch.utils.data.DataLoader(IndexedDataset(x), batch_size=bacth_size, shuffle=True, drop_last=True)
        
        while train_flag:
            if modified_em:
                self.sigmatilde, self.pitilde = self.maximization_step(x, self.expectation_step(x, leave_one_out=leave_one_out))
            else:
                for x_batch, idx in dataloader:
                    optimizer.zero_grad()
                    loss = -self(x_batch, leave_one_out=leave_one_out, batch_idx=idx).mean()
                    loss.backward()
                    optimizer.step()

            with torch.no_grad(): objective = self(x, leave_one_out=leave_one_out).mean()
            if objective.isnan(): raise ValueError('Objective Value is NaN!')

            itx+=1
            if verbose and itx%verbose_freq==0: print('Iteration: %d, Objective Value: %.5f'%(itx, objective))
    
            if wait_convergence:
                delta_objective = torch.abs(objective-last_objective)
                last_objective = objective
                if delta_objective<objective_threshold: train_flag=False
            else:
                if itx>=num_iterations: train_flag=False

class PiKernelDensityEstimator(AdaptiveKernelDensityEstimator):
    def __init__(self, mu, sigma=0.1):
        super(PiKernelDensityEstimator,self).__init__(mu, sigma)
    
    def _diff_parameterize(self):
        super()._diff_parameterize()
        self.pitilde = torch.nn.Parameter(self.pitilde)
    
    def maximization_step(self, x, responsibility_matrix):
        return super().maximization_step(x, responsibility_matrix)[0], from_pi(responsibility_matrix.sum(0,keepdim=True)/responsibility_matrix.sum())

