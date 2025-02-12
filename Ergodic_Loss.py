from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import math
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

import math

def psi(t):
    """
    A smooth bump function on (0,1) defined by:
    """
    #if t <= 0 or t >= 1:
    #    return 0.0
    #else:
    return torch.exp(-1.0 / (t * (1 - t)))

def test_varphi(x, cell, k_max):
    """
    Computes the smooth bump function for the grid cell specified by 'cell'
    on [0,1]^2.
    Parameters:
      x: tuple (x1, x2), a point in [0,1]^2.
      delta: side length of each grid cell (assumed to divide 1 exactly).
      cell: tuple (i, j) with 0 <= i, j < 1/delta, representing the grid cell.
    
    The function is defined by:
    
        varphi_{i,j}(x) = psi((x1 - i*delta)/delta) * psi((x2 - j*delta)/delta)
    
    It is maximal at the center of the cell and vanishes at (or outside) the cell boundaries.
    """
    delta = 1 / (k_max)
    i, j = cell
    # Normalize coordinates relative to the cell A_{i,j} = [i*delta, (i+1)*delta) x [j*delta, (j+1)*delta)
    #x[:,0] = torch.sigmoid((x[:,0] - i*delta) / delta) - torch.sigmoid((x[:,0] - (i + 1)*delta) / delta)
    #x[:,1] = torch.sigmoid((x[:,1] - j*delta) / delta) - torch.sigmoid((x[:,1] - (j + 1)*delta) / delta)
    #print(x)
    eps = 1 /k_max
    sigma = delta / 3  # Spread of the Gaussian
    c_x, c_y = (i + 0.5) * delta, (j + 0.5) * delta  # Center of the bump
    # Compute Gaussian bumps for each coordinate
    return torch.exp(-((x[..., 0] - c_x) ** 2) / (2 * sigma ** 2)) * torch.exp(-((x[..., 1] - c_y) ** 2) / (2 * sigma ** 2))



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
        self.k_compare = k_max

        coeff_shape = [self.k_max for _ in range(len(self.L))]
        self.eps = 0.1
        #self.sigma = 0.05
        self.sigma = max((math.sqrt(-2*math.log(self.eps))* (self.k_max -1)*2)**(-1), 0.1)
        print("calulcated sigma:", self.sigma)
        self.coeffs_density = torch.ones(coeff_shape, device = self.device)
        self.density = density
        self.init_densities() ## find right init funciton
        self.init_mydensity(kwargs) ## initialize parameters
        self.normalization_factors = torch.zeros(coeff_shape, device = self.device) # h_k
        self.norm_weights = torch.zeros(coeff_shape, device = self.device) # Lambda_k
        self.compute_fourier_coefficients_density()
        #print(self.coeffs_density, "target distribution")
        self.control_energy_reg = control_energy_reg

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
            plt.figure(figsize=(10, 6))
            plt.hist(samples, bins=30, density=True, alpha=0.6, color='skyblue', label='Rejection Sampling')

            x_range = np.linspace(min(samples), max(samples), 1000)
            plt.plot(x_range, self.pdf(x_range), color='red', linewidth=2, label='True PDF')

            plt.xlabel('Sample Value')
            plt.ylabel('Density')
            plt.title('Histogram of Rejection Sampling vs. True PDF')
            plt.legend()
            plt.show()
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

    def varphi(self, x, cell):
        #eps = self.eps
        eps = 0
        i, j = cell
        c = torch.tensor([1 /(self.k_max - 1) * i,  1/(self.k_max - 1) * j])
        return (torch.max(torch.tensor(0.0), torch.exp(-((x - c)**2).sum(dim=-1) / (2 * self.sigma**2)) - eps) + eps) / (self.sigma)
        #return (torch.max(torch.tensor(0.0), torch.exp(-((x - self.c[i,j])**2).sum(dim=-1) / (2 * self.sigma**2)) - eps) + eps) / (self.sigma)

    def sampled_fourier(self, k):
        """
            Takes samples from pdf and estimates real part of characteristic function
            samples = [num sample, dim], (dim of our Random Vector)
        """
        #i,j = k
        #mask = (self.samples[:, 0] >= i / self.k_max ) & (self.samples[:, 0] <= (i +1) / self.k_max) & (self.samples[:, 1] >= (j) / self.k_max) & (self.samples[:, 1] <= (j +1) / self.k_max)
        #return mask.to(float).sum() / self.n_samples
        return self.varphi(self.samples, k).sum() / self.n_samples

        #return torch.pow(self.samples, k).prod(dim = 1).sum() / self.n_samples

    def compute_weight_norm(self, k):
        """ compute Lambda_k """
        return (1 + (k**2).sum())**((-1) * ((len(self.L) + 1) / 2))


    def fourier_basis(self,x, sets):
        """ 
        x: State at time t_k [Num_timesteps ,Batch_size, N_Agents, in_dim]
        k torch.tensor (in_dim)
        """
        #k = torch.tensor(sets, dtype = torch.float32)
        return self.varphi(x, sets)
        #k *= (torch.pi * (self.L)**(-1))
        #k = k.to(self.device)
        #print(x.shape, k.view(1,1,1,-1).shape)
        #return torch.pow(x, k).prod(dim = -1)
        #return (torch.cos(x * k.view(1,1,1,-1)).prod(dim = -1)) / self.normalization_factors[sets]

        
    def compute_fourier_coefficients_agents_at_time_t(self,x, sets):
        """
        x: State of Agents [Num_timesteps ,Batch_size, N_Agents, in_dim] 
        returns c_k coefficient at time t of shape [Batch_size]
        """
        # For now i just put as calculaated t 1s
        transform = self.fourier_basis(x,sets)
        c_k = transform.sum(dim=0).sum(dim=-1)
        return c_k / (self.N_Agents * self.n_timesteps) ##TODO add functionality to not use [0,1] as timeframe, i.e * t

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
            #k = torch.tensor(sets, dtype = torch.float32)
            #k *= torch.pi * (self.L)**(-1)
            self.coeffs_density[sets] = self.charcteristic_function(sets).real
            #self.norm_weights[sets] = self.compute_weight_norm(k)
            #self.normalization_factors[sets] = self.compute_normalization_constant(sets)

        ## Scale the \Lambda_k such that we assign different importance to coeffs, without decreasing the values to much! (important for penalty on leaving rect!)
        #self.norm_weights_scaled = (self.norm_weights - self.norm_weights.min()) / (self.norm_weights.max() - self.norm_weights.min())
        self.norm_weights_scaled = torch.ones_like(self.coeffs_density)
        #self.coeffs_density /= self.normalization_factors


    def forward(self,x, u = None):
        """
        x: State of Agents [Num_timesteps ,Batch_size, N_Agents, in_dim] 
        """
        Batch_size = x.shape[1]
        #self.c = torch.clamp(torch.rand(self.k_max, self.k_max, self.in_dim), min = self.sigma, max = 1 - self.sigma)
        #self.c = torch.rand(self.k_max, self.k_max, self.in_dim)
        #self.compute_fourier_coefficients_density()
        coeffs = torch.ones(([Batch_size] + [self.k_compare for _ in range(len(self.L))]), device = self.device)
        #if len(self.L) > 1:
        #    raise NotImplementedError('Only one dimension available so far...')
        #coeffs = self.compute_C_t_fft(x)
        k = list(range(self.k_compare))
        for sets in product(k, repeat = len(self.L)):
            slices = [slice(None)] + list(sets)
            coeffs[slices] = self.compute_fourier_coefficients_agents_at_time_t(x,sets)
        idx = [slice(self.k_compare) for _ in range(len(self.L))]
        
        #loss_2 = (((coeffs - self.coeffs_density)**2) * self.norm_weights).sum()
        #loss_1 = (((coeffs - self.coeffs_density).abs()) * self.norm_weights).sum()
        #loss = lam2 * loss_2 + lam1 * loss_1
        #repeated_coeffs = self.coeffs_density[idx].unsqueeze(0).repeat(Batch_size, *[1 for _ in self.L]) ##repeat target coeffs Batch_size times
        #loss = self.criterion(self.norm_weights * coeffs, self.norm_weights * repeated_coeffs)
        ## Elastic net over coeffs
        #print(repeated_coeffs.shape)
        loss = ((self.norm_weights_scaled[idx] * (coeffs - self.coeffs_density[idx])).pow(2).sum(dim = 1) / self.norm_weights_scaled[idx].sum()).mean() ## some sort of elastic Net
 ## some sort of elastic Net
        #print((self.norm_weights * (coeffs - repeated_coeffs)).abs().sum(dim = 1))

        loss += ((self.norm_weights_scaled[idx] * (coeffs - self.coeffs_density[idx])).abs().sum(dim=tuple(range(1, coeffs.ndim))) / self.norm_weights_scaled[idx].sum()).mean()
        loss /= 2
        ## some sort of elastic Net,normalized with weighted sum
        #print(loss, "current loss")
        ##### Keep in mind in conjunction with the model penalty which is just the amount of overstepping we need to have this normalized in a decent range!, both errors should be kind of proportional
        if self.verbose:
            print(coeffs.shape)
            print("model:", coeffs,"target:", self.coeffs_density)
            #print(((coeffs - repeated_coeffs)).abs().mean() / 2)
            #print(coeffs - repeated_coeffs, "difference")
            #print((coeffs - repeated_coeffs))
            #print("model", self.norm_weights_scaled * coeffs,"target", self.norm_weights_scaled * self.coeffs_density)
            #print("scaling", self.norm_weights_scaled, self.norm_weights_scaled)
        if u is not None:
            #TODO
            loss += (self.control_energy_reg * (u.abs() ** 2).mean()) / 2#(2 * self.N_Agents * self.n_timesteps * Batch_size) ### minimize control energy, w.r.t L2 norm squared 
        return  loss## I am really unhappy with the expand here!


if __name__ == '__main__':
    import time
    start_time = time.time()
    N_Agents = 2
    num_timesteps = 2
    batch_size = 1
    device = torch.device("cpu")

    from densities import uniform_rect_regions as pdf
    import functools
#    region  = np.array([[[0, 1.], [0, 1.]],
#                        [[0.6, 0.9], [0.7, 0.9]]])
    region  = np.array([[[0, 1.], [0, 1.]],
])
    custom_pdf = functools.partial(pdf, regions=region)
    in_dim = 2 
    Loss = Ergodicity_Loss(N_Agents, num_timesteps, in_dim = in_dim, k_max = 3, device = device, density = 'custom', pdf = custom_pdf,max_pdf = 5/3, num_samples = 1000)
    print("sampled same distribution",Loss.coeffs_density)
    print(Loss.normalization_factors, "normals")
    Loss.verbose = True
    X = torch.randn([num_timesteps,batch_size,N_Agents,in_dim], requires_grad = True, device = device)
    print(Loss(X))
    Loss.k_compare = 3
    print(Loss(X))

