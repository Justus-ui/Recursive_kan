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
        self.coeffs_density = torch.zeros(coeff_shape, device = self.device)
        self.density = density
        self.init_densities() ## find right init funciton
        self.init_mydensity(kwargs) ## initialize parameters
        self.normalization_factors = torch.zeros(coeff_shape, device = self.device) # h_k
        self.norm_weights = torch.zeros(coeff_shape, device = self.device) # Lambda_k
        self.compute_fourier_coefficients_density()
        print(self.coeffs_density, "target distribution")

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
            self.norm_weights[sets] = self.compute_weight_norm(k)
            self.normalization_factors[sets] = self.compute_normalization_constant(sets)
            if all(x == 0 for x in sets):
                self.norm_weights[sets] = 0. 
        ## Scale the \Lambda_k such that we assign different importance to coeffs, without decreasing the values to much! (important for penalty on leaving rect!)
        self.norm_weights_scaled = (self.norm_weights - self.norm_weights.min()) / (self.norm_weights.max() - self.norm_weights.min())
        self.coeffs_density /= self.normalization_factors


    def forward(self,x, u = None):
        """
        x: State of Agents [Num_timesteps ,Batch_size, N_Agents, in_dim] 
        """
        Batch_size = x.shape[1]
        coeffs = torch.zeros(([Batch_size] + [self.k_compare for _ in range(len(self.L))]), device = self.device)
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
            print("model:", coeffs[0,1],"target:", self.coeffs_density[1])
            #print(((coeffs - repeated_coeffs)).abs().mean() / 2)
            #print(coeffs - repeated_coeffs, "difference")
            #print((coeffs - repeated_coeffs))
            #print("model", self.norm_weights_scaled * coeffs,"target", self.norm_weights_scaled * self.coeffs_density)
            #print("scaling", self.norm_weights_scaled, self.norm_weights_scaled)
        if u is not None:
            #TODO
            loss += self.control_energy_reg * 1e2 * ((u[:5,:,:,:].abs()**2).mean()) ### make sure at the beginning small steps!
            loss += (self.control_energy_reg * (u[5:,:,:,:].abs() ** 2).mean()) / 2#(2 * self.N_Agents * self.n_timesteps * Batch_size) ### minimize control energy, w.r.t L2 norm squared 
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

    def custom_pdf(x):
        return np.where(((x > 0) & (x < 0.3)) | ((x > 0.6) & (x < 0.9)), 5 / 3, 0)
    Loss = Ergodicity_Loss(N_Agents, num_timesteps, in_dim = in_dim, k_max = 12, device = device, density = 'custom', pdf = custom_pdf,max_pdf = 5/3, num_samples = 1000)
    print("sampled same distribution",Loss.coeffs_density)
    print(Loss.normalization_factors, "normals")
    X = torch.randn([num_timesteps,batch_size,N_Agents,in_dim], requires_grad = True, device = device)
    print(Loss(X))
    Loss.k_compare = 6
    print(Loss(X))

    from densities import uniform_rect_regions as pdf
    import functools
    region  = np.array([[[0, 0.3], [0, 0.3]],
                        [[0.6, 0.9], [0.7, 0.9]]])
    custom_pdf = functools.partial(pdf, regions=region)
    in_dim = 2 
    Loss = Ergodicity_Loss(N_Agents, num_timesteps, in_dim = in_dim, k_max = 12, device = device, density = 'custom', pdf = custom_pdf,max_pdf = 5/3, num_samples = 1000)
    print("sampled same distribution",Loss.coeffs_density)
    print(Loss.normalization_factors, "normals")
    X = torch.randn([num_timesteps,batch_size,N_Agents,in_dim], requires_grad = True, device = device)
    print(Loss(X))
    Loss.k_compare = 6
    print(Loss(X))

