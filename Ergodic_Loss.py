from itertools import product

import numpy as np
from scipy.integrate import nquad

import torch
from torch import nn

class Ergodicity_Loss(nn.Module):
    def __init__(self, N_Agents, n_timesteps,L = None, in_dim = None, k_max = 10, control_energy_reg = 1e-3, device = torch.device('cpu'), density = 'uniform', **kwargs):
        super(Ergodicity_Loss, self).__init__()
        self.device = device
        #self.device = torch.device("cpu")
        self.N_Agents = N_Agents
        self.n_timesteps = n_timesteps
        if L is None:
            self.L = torch.tensor([1. for _ in range(in_dim)])
        else:
            self.L = L
        self.k_max = k_max

        coeff_shape = [self.k_max for _ in range(len(self.L))]
        self.coeffs_density = torch.zeros(coeff_shape, device = self.device)
        self.density = density
        self.init_densities() ## find right init funciton
        self.init_mydensity(kwargs) ## initialize parameters
        self.normalization_factors = torch.zeros(coeff_shape, device = self.device) # h_k
        self.norm_weights = torch.zeros(coeff_shape, device = self.device) # Lambda_k
        self.compute_fourier_coefficients_density()

        self.control_energy_reg = control_energy_reg

    def type_error(self):
        raise TypeError('Unknow density')

    def init_densities(self):
        def noop(*args, **kwargs):
            pass
        init_params_densities = {
            'uniform': noop,
            'mixture_uniform': self.init_mixture_uniform
        }
        self.init_mydensity = init_params_densities.get(self.density, self.type_error)

        self.init = init_params_densities.get(self.density, self.type_error)
        cf_densities = {
            'uniform': self.charcteristic_function_uniform,
            'mixture_uniform': self.characteristic_function_mixture_uniform
        }
        self.charcteristic_function = cf_densities.get(self.density, self.type_error)
    
    def init_mixture_uniform(self, kwargs):
        self.regions = kwargs['regions']
        self.weights = kwargs['weights']

    def compute_normalization_constant(self,k):
        """ 
        h_k 
        """
        for i, value in enumerate(k):
            normal = 1.
            if value == 0:
                continue
            else:
                normal *= (self.L[i] / 2)**0.5
        return normal

    def compute_weight_norm(self, k):
        """ compute Lambda_k """
        return (1 + (k**2).sum())**((-1) * ((len(self.L) + 1) / 2))


    def fourier_basis(self,x, sets):
        """ 
        x: State at time t_k [Num_timesteps ,Batch_size, N_Agents, in_dim]
        k torch.tensor (in_dim)
        """
        k = torch.tensor(sets, dtype = torch.float32)
        k *= (torch.pi * (self.L)**(-1))
        k = k.to(self.device)
        #print(x.shape, k.view(1,1,1,-1).shape)
        return (torch.cos(x * k.view(1,1,1,-1)).prod(dim = -1)) / self.normalization_factors[sets]

        
    def compute_fourier_coefficients_agents_at_time_t(self,x, sets):
        """
        x: State of Agents [Num_timesteps ,Batch_size, N_Agents, in_dim] 
        returns c_k coefficient at time t of shape [Batch_size]
        """
        # For now i just put as calculaated t 1s
        transform = self.fourier_basis(x,sets)
        c_k = transform.sum(dim=-3).sum(dim=-1)
        return c_k / (self.N_Agents * self.n_timesteps)

    def charcteristic_function_uniform(self, k):
        """ 
            Characteristic Function of uniform distirbution over [0,L1] x [0,L2] x ... x [0,Ln]
            k tuple of ints
            returns
            k-th coefficient of charcteristic function!
        """
        coeff = torch.ones(1, dtype=torch.complex64)
        for i,Li in enumerate(self.L):
            Li = self.L[i]
            ki = k[i]
            if ki != 0:
                integral = (torch.exp(1j * ki * Li) - 1) / (1j * ki)
            else:
                integral = Li
            coeff *= integral / Li
        return coeff

    def characteristic_function_mixture_uniform(self, k):
        """
        Compute the characteristic function of a random vector X = (X1, X2, ..., Xn)
        uniformly distributed over multiple regions A1, A2, ..., Am using PyTorch.
        """

        phi = 0.0 + 0.0j
        norm = 0.0 + 0.0j
        for i, region in enumerate(self.regions):
            region_phi = 1.0 + 0.0j
            for j,interval in enumerate(region):
                a_j, b_j = interval
                k_j = k[j]
                if k_j != 0:
                    region_phi *= (torch.exp(1j * k_j * b_j) - torch.exp(1j * k_j * a_j)) / (1j * k_j * (b_j - a_j))
                else:
                    region_phi *= (b_j - a_j)
                volume = (b_j - a_j) ## TODO measure of set greater!
            norm += (volume * self.weights[i])
            phi += (self.weights[i] * region_phi)
        return (phi / norm).to(self.device)

    def compute_fourier_coefficients_density(self):
        k = list(range(self.k_max))
        for sets in product(k, repeat = len(self.L)):
            k = torch.tensor(sets, dtype = torch.float32)
            k *= torch.pi * (self.L)**(-1)
            self.coeffs_density[sets] = self.charcteristic_function(k).real
            self.normalization_factors[sets] = self.compute_normalization_constant(sets)
            self.norm_weights[sets] = self.compute_weight_norm(k)
        self.coeffs_density /= self.normalization_factors


    def forward(self,x, u = None):
        """
        x: State of Agents [Num_timesteps ,Batch_size, N_Agents, in_dim] 
        """
        Batch_size = x.shape[1]
        coeffs = torch.zeros(([Batch_size] + [self.k_max for _ in range(len(self.L))]), device = self.device)
        k = list(range(self.k_max))
        for sets in product(k, repeat = len(self.L)):
            slices = [slice(None)] + list(sets)
            coeffs[slices] = self.compute_fourier_coefficients_agents_at_time_t(x,sets)
        
        loss = (((coeffs - self.coeffs_density)**2) * self.norm_weights).sum()
        if u is not None:
            loss += (self.control_energy_reg * (u.abs() ** 2).sum()) / (2 * self.N_Agents * self.n_timesteps * Batch_size) ### minimize control energy, w.r.t L2 norm squared 
        return  loss## I am really unhappy with the expand here!

if __name__ == '__main__':
    import time
    start_time = time.time()
    N_Agents = 2
    num_timesteps = 100
    batch_size = 32
    in_dim = 1
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    X = torch.randn([num_timesteps,batch_size,N_Agents,in_dim], requires_grad = True, device = device)
    Loss = Ergodicity_Loss(N_Agents, num_timesteps, in_dim = in_dim, k_max = 64, device = device, density = 'uniform')
    print(Loss.coeffs_density)
    intermed_time = time.time()
    print("init time:", intermed_time- start_time)
    print(Loss(X))
    end_time = time.time()
    forward_time = end_time - intermed_time
    print("forward_time:", forward_time)

    regions = [
            torch.tensor([[.0, .3]]),
            torch.tensor([[.6, .9]])
            ]
    weights = [.5, .5]
    Loss = Ergodicity_Loss(N_Agents, num_timesteps, in_dim = in_dim, k_max = 64, device = device, density = 'mixture_uniform', regions = regions, weights = weights)
    print(Loss.coeffs_density)
    intermed_time = time.time()
    print("init time:", intermed_time- start_time)
    print(Loss(X))
    end_time = time.time()
    forward_time = end_time - intermed_time
    print("forward_time:", forward_time)

