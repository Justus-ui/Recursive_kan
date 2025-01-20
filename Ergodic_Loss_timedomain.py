from itertools import product

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn


class Ergodicity_Loss(nn.Module):
    def __init__(self, N_Agents, n_timesteps,L = None, in_dim = None, control_energy_reg = 1e-3, device = torch.device('cpu'), density = 'uniform', verbose = True, **kwargs):
        super(Ergodicity_Loss, self).__init__()
        self.device = device
        self.verbose = verbose
        #self.device = torch.device("cpu")
        self.N_Agents = N_Agents
        self.n_timesteps = n_timesteps
        if L is None:
            self.L = torch.tensor([1. for _ in range(in_dim)])
            self.in_dim = in_dim
        else:
            self.L = L
            self.in_dim = len(L)
        self.density = density
        self.init_densities() ## find right init funciton
        self.init_mydensity(kwargs) ## initialize parameters

        self.control_energy_reg = control_energy_reg
        sample_eps = 1e-5 ## avoid instability around edges!
        self.samples = torch.linspace(sample_eps, self.L[0] - sample_eps, self.n_samples)
        self.targets = self.pdf(self.samples)


    def type_error(self):
        raise TypeError('Unknow density')

    def init_densities(self):
        """
            Initalizes densities
        """
        def noop(*args, **kwargs):
            pass
        init_params_densities = {
            'uniform': noop,
            'mixture_uniform': self.init_mixture_uniform,
            'custom': self.init_custom_pdf
        }
        self.init_mydensity = init_params_densities.get(self.density, self.type_error)

    def init_custom_pdf(self, kwargs):
        self.pdf = kwargs['pdf']
        self.n_samples = kwargs['num_samples']

    def init_mixture_uniform(self, kwargs):
        self.regions = kwargs['regions']
        self.weights = kwargs['weights']

    def C_t(self,samples, X):
        """
            X = [Num_timesteps ,Batch_size, N_Agents, in_dim]
            x = linspace over domain for now R^{1}
            C_t in one dimension
            This is the distribution induced by the current trajectory, we try to minimize distance to wanted trajectory -> However Sobolev space representation was deemed no usefull as computation of Fourier coeffs harder than in original paper
        """
        eps = 1e-10 # Avoid division by 0 error --> Approximation of real function can be neglected and samples resulting in nan may be ignored...
        alpha = 100
        arr = X - samples
        #normalized = arr / (arr.abs() + eps)
        #normalized = arr / (arr.pow(2).sum().sqrt() + eps)
        #print(normalized)
        normalized = torch.tanh(alpha * arr)
        diff = torch.diff(normalized, dim = 0)
        sign_changes = diff.abs().sum(dim=(0,2)) / 2
        #print(sign_changes.shape)
        #print(sign_changes.mean(dim = 1).shape)
        return (sign_changes) / (sign_changes.mean(dim = 1).unsqueeze(1) + eps)



    def forward(self,x, u = None):
        """
        x: State of Agents [Num_timesteps ,Batch_size, N_Agents, in_dim] 
        """
        Batch_size = x.shape[1]
        if len(self.L) > 1:
            raise NotImplementedError('Only one dimension available so far...')
                
        function_values = self.C_t(self.samples, x)
        repeated_targets = self.targets.unsqueeze(0).repeat(Batch_size, *[1 for _ in self.L]) ##repeat target coeffs Batch_size times
        crit = nn.L1Loss(reduction='mean')
        loss = crit(function_values, repeated_targets)
        self.function_values = function_values
        #loss += 0.5 * torch.max(torch.abs(function_values - repeated_targets), dim = 1)[0].mean() ### || . ||_inf as it is tightest. My guess not stable -> bigger issue non stability of parameterised C_t
        if self.verbose:
            print("model", function_values[0],"target", self.targets)
            print(torch.max(torch.abs(function_values - repeated_targets), dim = 1)[0].mean(), "loss")
        if u is not None:
            loss += (self.control_energy_reg * (u.abs() ** 2).sum()) / (2 * self.N_Agents * self.n_timesteps * Batch_size) ### minimize control energy, w.r.t L2 norm squared 
        return loss   #reduction = 'mean'


if __name__ == '__main__':
    import time
    start_time = time.time()
    N_Agents = 2
    num_timesteps = 100
    batch_size = 32
    in_dim = 1
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    X = torch.rand([num_timesteps,batch_size,N_Agents,in_dim], requires_grad = True, device = device)
    def custom_pdf(x):
        return torch.tensor(np.where(((x > 0) & (x < 0.3)) | ((x > 0.6) & (x < 0.9)), 5 / 3, 0))
    Loss = Ergodicity_Loss(N_Agents, num_timesteps, in_dim = 1, device = device, density = 'custom', pdf = custom_pdf, num_samples = 1000)
    print(Loss(X))


