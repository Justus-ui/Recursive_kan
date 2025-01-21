from itertools import product

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

### draw_samples from pdf
def rejection_sampling(pdf, in_dim, n_samples, L, max_pdf_value=1.0):
    """
    Rejection sampling to draw samples from a given PDF.
    
    Parameters:
    - pdf: The custom PDF function.
    - n_samples: Number of samples to generate.
    - x_range: Range of x values to sample from.
    - max_pdf_value: The maximum value of the PDF to normalize the proposal distribution.
    
    Returns:
    - samples: List of samples drawn from the PDF.
    """
    samples = []
    while len(samples) < n_samples:
        # Generate random candidate sample from the proposal distribution (uniform)
        x_candidate = np.array([np.random.uniform(0, upper, size=(1)) for upper in L]).T
        y_candidate = np.random.uniform(0, max_pdf_value)
        
        # Accept the sample with probability proportional to the PDF value
        if y_candidate < pdf(x_candidate):
            samples.append(x_candidate)
    return torch.tensor(np.array(samples))

def differentiable_sampling(pdf, n_samples, x_range = (1e-2,1 - 1e-2), max_pdf = 11):
    # Assuming the pdf is normalized so that it integrates to 1 over the range
    samples = []
    gen_samples = 0
    while gen_samples < n_samples:
        # Sample x_candidate from the range
        x_candidate = torch.rand(1) * (x_range[1] - x_range[0]) + x_range[0]
        
        # Sample y_candidate from a uniform distribution between 0 and 1
        y_candidate = torch.rand(1) * max_pdf
        
        # Compute the probability of acceptance as a soft threshold
        acceptance_probability = torch.sigmoid(pdf(x_candidate) - y_candidate)  # smooth function to decide acceptance
        try:
            accept = torch.bernoulli(acceptance_probability)
        except:
            #print(x_candidate)
            ## sometimes pdf generates division by 0 error --> nan
            continue
        
        # Only append x_candidate if accepted, but using a differentiable approximation
        if accept.item(): 
            sample = x_candidate * accept
            #print(accept, x_candidate)
            samples.append(sample)
            gen_samples += 1
    
    return torch.cat(samples)

class Ergodicity_Loss(nn.Module):
    def __init__(self, N_Agents, n_timesteps,L = None, in_dim = None, k_max = 10, control_energy_reg = 1e-3, device = torch.device('cpu'), density = 'uniform', verbose = True, **kwargs):
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
        self.k_max = k_max

        coeff_shape = [self.k_max for _ in range(len(self.L))]
        self.coeffs_density = torch.zeros(coeff_shape, device = self.device)
        self.density = density
        self.init_densities() ## find right init funciton
        self.init_mydensity(kwargs) ## initialize parameters
        self.normalization_factors = torch.zeros(coeff_shape, device = self.device) # h_k
        self.norm_weights = torch.zeros(coeff_shape, device = self.device) # Lambda_k
        self.compute_fourier_coefficients_density()
        print(self.coeffs_density, "target distribution")

        self.control_energy_reg = control_energy_reg
        #self.criterion = torch.nn.SmoothL1Loss(beta = 1e-3)
        self.criterion = torch.nn.MSELoss()

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

        self.init = init_params_densities.get(self.density, self.type_error)
        cf_densities = {
            'uniform': self.charcteristic_function_uniform,
            'mixture_uniform': self.characteristic_function_mixture_uniform,
            'custom': self.sampled_fourier
        }
        self.charcteristic_function = cf_densities.get(self.density, self.type_error)

    def init_custom_pdf(self, kwargs):
        self.pdf = kwargs['pdf']
                # Parameters
        self.n_samples = kwargs['num_samples']
        max_pdf_value = kwargs['max_pdf']

        # Generate samples using rejection sampling
        samples = rejection_sampling(self.pdf, self.in_dim, self.n_samples, self.L, max_pdf_value)
        samples = samples.squeeze()
        if len(samples.shape) == 1:
            samples = samples.unsqueeze(1)
        
        if self.in_dim == 2 and self.verbose:
            # Plotting the samples and the true PDF (in 2D)
            Z = self.pdf(samples.numpy())
            print(Z.shape)
            plt.scatter(samples[:, 0], samples[:, 1], c=Z, cmap="viridis")
            plt.colorbar(label="Function Value")
            plt.legend()
            plt.title("Rejection Sampling from a Custom PDF (2D)")
            plt.show()
        self.samples = samples

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

    def sampled_fourier(self, k):
        """
            Takes samples from pdf and estimates real part of characteristic function
            samples = [num sample, dim], (dim of our Random Vector)
        """

        return torch.cos(self.samples * k).prod(dim = 1).sum() / self.n_samples






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
        c_k = transform.sum(dim=0).sum(dim=-1)
        return c_k / (self.N_Agents * self.n_timesteps) ##TODO add functionality to not use [0,1] as timeframe, i.e * t
    
    def C_t(self, x):
        """
            C_t in one dimension 
            TODO VECTTORIZE
        """

        eps = 1e-6 ## TODO
        arr = self.Curr_sample - x
        normalized = arr / (arr.abs())
        diff = torch.diff(normalized, dim = 0)
        return diff.abs().sum(dim=(0,2)) / 2 

    def compute_C_t_fft(self, X):
        """
            C_t in one dimension 
            TODO VECTTORIZE
            X State of Agents [Num_timesteps ,Batch_size, N_Agents, in_dim] 
        """
        n_samples = 1000 
        coeff_shape = [X.shape[1]] + [self.k_max for _ in range(len(self.L))]
        coeffs_density = torch.zeros(coeff_shape)
        #print(coeffs_density.shape)
        k = list(range(self.k_max))
        for i in range(X.shape[1]):
            self.Curr_sample = X[:,i,:,:].unsqueeze(1)
            samples = differentiable_sampling(self.C_t, n_samples)
            for sets in product(k, repeat = len(self.L)):
                k_trans = torch.tensor(sets, dtype = torch.float32)
                k_trans *= torch.pi * (self.L)**(-1)
                coeffs_density[i,*sets] = (torch.cos(samples * k_trans).prod(dim = 1).sum() / n_samples) / self.normalization_factors[sets]
        return coeffs_density


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
            self.norm_weights[sets] = self.compute_weight_norm(k) * 100
        self.coeffs_density /= self.normalization_factors


    def forward(self,x, u = None):
        """
        x: State of Agents [Num_timesteps ,Batch_size, N_Agents, in_dim] 
        """
        Batch_size = x.shape[1]
        coeffs = torch.zeros(([Batch_size] + [self.k_max for _ in range(len(self.L))]), device = self.device)
        #if len(self.L) > 1:
        #    raise NotImplementedError('Only one dimension available so far...')
        #coeffs = self.compute_C_t_fft(x)
        k = list(range(self.k_max))
        for sets in product(k, repeat = len(self.L)):
            slices = [slice(None)] + list(sets)
            coeffs[slices] = self.compute_fourier_coefficients_agents_at_time_t(x,sets)
        
        #loss_2 = (((coeffs - self.coeffs_density)**2) * self.norm_weights).sum()
        #loss_1 = (((coeffs - self.coeffs_density).abs()) * self.norm_weights).sum()
        #loss = lam2 * loss_2 + lam1 * loss_1
        repeated_coeffs = self.coeffs_density.unsqueeze(0).repeat(Batch_size, *[1 for _ in self.L]) ##repeat target coeffs Batch_size times
        #loss = self.criterion(self.norm_weights * coeffs, self.norm_weights * repeated_coeffs)
        loss = ((self.norm_weights * (coeffs - repeated_coeffs))**2).mean()
        if self.verbose:
            print("model", self.norm_weights * coeffs,"target", self.norm_weights * self.coeffs_density)
            print("scaling", self.norm_weights)
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
    Loss = Ergodicity_Loss(N_Agents, num_timesteps, in_dim = in_dim, k_max = 4, device = device, density = 'uniform')
    print(Loss.coeffs_density)
    print(Loss.coeffs_density.shape)
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
    Loss = Ergodicity_Loss(N_Agents, num_timesteps, in_dim = in_dim, k_max = 4, device = device, density = 'mixture_uniform', regions = regions, weights = weights)
    print("uniform via cf", Loss.coeffs_density)
    print(Loss.coeffs_density.shape)
    intermed_time = time.time()
    print("init time:", intermed_time- start_time)
    print(Loss(X))
    end_time = time.time()
    forward_time = end_time - intermed_time
    print("forward_time:", forward_time)
    #from densities import uniform_rect_regions as pdf
    #import functools
    #region  = np.array([[[0, 0.3], [0, 0.3]],
    #                    [[0.6, 0.9], [0.7, 0.9]]])
    #custom_pdf = functools.partial(pdf, regions=regions)
    def custom_pdf(x):
        return np.where(((x > 0) & (x < 0.3)) | ((x > 0.6) & (x < 0.9)), 5 / 3, 0)

    Loss = Ergodicity_Loss(N_Agents, num_timesteps, in_dim = 1, k_max = 12, device = device, density = 'custom', pdf = custom_pdf,max_pdf = 5/3, num_samples = 100000)
    print("sampled same distribution",Loss.coeffs_density)
    print(Loss.normalization_factors, "normals")


