import torch
import torch.nn as nn

from Lipschitz_Linear import Lipschitz_Linear
from Lipschitz_GRU import Lipschitz_GRU

class KAN_RNN_Layer(nn.Module):
    """
        Defines KAN with univariate approximation via GRUs
    """
    def __init__(self, N_Agents, in_dim, hidden, depth, n_timesteps, sys_param_lam = 0.1, u_max = 5, network_type = 'multi', thres = 0., device = None, dropout = 0):
        """ 
        in_dim:Dimension of Agent information, i.e cartesian coordinates R^2
        device: if true uses cuda
        """
        super(KAN_RNN_Layer, self).__init__()
        # Problem Attributes
        self.N_Agents = N_Agents
        self.in_dim = in_dim
        # Model attributes
        self.dropout = dropout
        self.hidden = hidden
        self.depth = depth
        self.sys_param_lam = sys_param_lam
        self.device_train = torch.device("cpu") ## if device marker is set --> cuda
        if device:
            assert torch.cuda.is_available()
            self.device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device_eval = torch.device("cpu")
        self.device_eval = torch.device("cpu")
        self.network_type = network_type
        self.Network_stack = nn.ModuleList() ## Read linear_Network_stack["to Neuron"]["from Neuron"]
        self.linear_Network_stack = nn.ModuleList()
        self.activation = nn.ReLU()
        self.num_forward_steps = n_timesteps
        self.u_max = u_max
        self.SNR = 3 #### actual signal / Noise = 4, no DB

        # Constraint
        self.thres = thres
        self.L = [1. for _ in range(in_dim)] ## for projecting into bounds if Agent tries to leave rect #TODO write general
        
        #init model
        self.init_model()

    def train(self, mode = True):
        super(KAN_RNN_Layer, self).train(mode)
        self.to_device()

    def eval(self):
        super(KAN_RNN_Layer, self).eval()
        self.to_device()

    def to_device(self):
        """
            Loads modules to device
        """
        device = self.device_train if self.training else self.device_eval
        #print(f"loaded to device {device}")
        self.to(device)

    def type_error(self):
        """
            Error on unknown networktype
        """
        raise TypeError(f"{self.network_type} is a invalid Networktype")

    def init_model(self):
        init_methods = {
            'multi': self.init_layers_multi_var,
            'uni': self.init_layers_uni_var
        }
        init_action = init_methods.get(self.network_type, self.type_error)
        init_action()

        forward_methods = {
            'multi': self.forward_multi,
            'uni': self.forward_uni
        }
        self.forward_action = forward_methods.get(self.network_type, self.type_error)


    def init_layers_multi_var(self):
        """
            defines the multivariate grus if we estimate f: R^(in_dim) -> R
        """
        for _ in range(self.N_Agents * self.in_dim):
            Networks = nn.ModuleList()
            for _ in range(self.N_Agents):
                Networks.append(Lipschitz_GRU(in_dim = self.in_dim ,hidden = self.hidden, depth = self.depth, dropout = self.dropout)) ## Dimension of input x -> indim, depth number of stacked Gru Layers, hidden Numer of Neurons in the GRu Layers
            self.Network_stack.append(Networks)
            self.linear_Network_stack.append(Lipschitz_Linear([self.N_Agents * self.hidden, 1]))
    
    def init_layers_uni_var(self):
        """
            defines the univariate grus if we estimate f: R^ -> R
        """
        for _ in range(self.N_Agents * self.in_dim):
            Networks = nn.ModuleList()
            for _ in range(self.N_Agents * self.in_dim):
                Networks.append(Lipschitz_GRU(in_dim = 1 ,hidden = self.hidden, depth = self.depth, dropout = self.dropout))
            self.Network_stack.append(Networks)
            self.linear_Network_stack.append(Lipschitz_Linear([self.N_Agents * self.hidden * self.in_dim, 1]))

    def time_step_multi(self, x):
        """
        When Networktype is multi
        x: Inital States [Batch_size, N_Agents, in_dim]
        """
        outs = torch.zeros_like(x)
        for i in range(self.N_Agents): ## out
            for l in range(self.in_dim):
                output_list = []
                for j in range(self.N_Agents): ### in
                    output_list.append(self.Network_stack[i * self.in_dim + l][j](x[:,j,:].unsqueeze(1)))
                out = self.linear_Network_stack[i * self.in_dim + l](self.activation(torch.cat(output_list, dim=1).reshape(-1, self.N_Agents * self.hidden)))
                #outs[:,i,:] =  ## Use this to have max energy constraint!
                outs[:,i,l] = torch.clamp(out.squeeze(), min = -self.u_max, max = self.u_max)
        #outs += torch.randn_like(outs)
        return outs

    def system_dynamics_multi(self,u, x_prev):
        """
            When Networktype is multi
            x_prev = [Batch_size, N_Agents, in_dim]
        """
        x_new = torch.zeros_like(x_prev)
        for i in range(self.in_dim):
            x_state = (x_prev + self.sys_param_lam * u)[:,:,i]
            x_new[:,:,i] = torch.clamp(x_state, min = 0, max = self.L[i])
            if self.training:
                self.penalty += torch.sum((torch.relu(-x_state - self.thres))**2) + torch.sum((torch.relu(x_state - self.L[i] - self.thres))**2)
        return x_new
    
    def forward_multi(self, x):
        """
        x: Inital States [Batch_size, N_Agents, in_dim] 

        return 
        control_trajectory: controller output [Num_timesteps ,Batch_size, N_Agents, in_dim]
        outs: State of Agents [Num_timesteps ,Batch_size, N_Agents, in_dim]  

        """ 
        self.penalty = 0
        device = self.device_train if self.training else self.device_eval
        outs = torch.zeros(self.num_forward_steps, *x.shape, device = device)
        control_trajectory = torch.zeros(self.num_forward_steps, *x.shape, device = device) ## Assume u of same shape as x!
        for i in range(self.num_forward_steps):
            outs[i] = x
            u = self.time_step_multi(x)
            x = self.system_dynamics_multi(u, x)
            control_trajectory[i] = u
        return outs, control_trajectory

    def time_step_uni(self, x):
        """
        When Networktype is uni
        x: Inital States [Batch_size, N_Agents, in_dim]
        """
        distances = self.get_distances_matrix(x)
        outs = torch.zeros_like(x)
        for i in range(self.N_Agents): ## out
            for l in range(self.in_dim):
                output_list = []
                for j in range(self.N_Agents): ### in
                    for k in range(self.in_dim):
                        output_list.append(self.Network_stack[i * self.in_dim + l][j * self.in_dim + k](x[:,j,k].unsqueeze(1).unsqueeze(1))) ## TODO made some changes here #### 
                out = self.linear_Network_stack[i * self.in_dim + l](self.activation(torch.cat(output_list, dim=1).reshape(-1, self.N_Agents * self.hidden * self.in_dim)))
                #print(i * self.in_dim + l, self.activation(torch.cat(output_list, dim=1).reshape(-1, self.N_Agents * self.hidden * self.in_dim)), self.hidden)
            #outs[:,i,:] = torch.clamp(out, min = -self.u_max, max = self.u_max) ## Use this to have max energy constraint!
                outs[:,i,l]  = torch.clamp(out.squeeze(), min = -self.u_max, max = self.u_max)
        #noise = torch.randn_like(outs)
        #scaled_noise = noise * (outs / self.SNR)
        #outs += scaled_noise
        return outs
    
    def get_distances_matrix(self, x):
        ##### returns fading of chanel at curent time!
        x_expanded = x.unsqueeze(2) - x.unsqueeze(1)
        distances = torch.sqrt(torch.sum(x_expanded ** 2, dim=-1))
        return (distances + 1e-1)**(-2) * torch.sqrt((torch.tensor(self.L)**2).sum())
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask = mask.triu(diagonal=1) 
        return distances.masked_select(mask).view(batch_size, -1)

    def system_dynamics_uni(self,u, x_prev, train = False):
        """
            When Networktype is uni
            x_prev = [Batch_size, N_Agents, in_dim]
        """
        x_new = torch.zeros_like(x_prev)
        for i in range(self.in_dim):
            x_state = (x_prev + self.sys_param_lam * u)[:,:,i]
            x_new[:,:,i] = torch.clamp(x_state, min = 0, max = self.L[i])
            if self.training:
                self.penalty += torch.sum((torch.relu(-x_state - self.thres))**2) + torch.sum((torch.relu(x_state - self.L[i] - self.thres))**2)
        return x_new
    
    def forward_uni(self, x):
        """
        x: Inital States [Batch_size, N_Agents, in_dim] 

        return 
        control_trajectory: controller output [Num_timesteps ,Batch_size, N_Agents, in_dim]
        outs: State of Agents [Num_timesteps ,Batch_size, N_Agents, in_dim]  

        """ 
        device = self.device_train if self.training else self.device_eval
        outs = torch.zeros(self.num_forward_steps, *x.shape, device = device)
        self.penalty = 0
        control_trajectory = torch.zeros(self.num_forward_steps, *x.shape, device = device) ## Assume u of same shape as x!
        for i in range(self.num_forward_steps):
            outs[i] = x
            u = self.time_step_uni(x)
            x = self.system_dynamics_uni(u, x)
            control_trajectory[i] = u
        return outs, control_trajectory

    def forward(self, x):
        device = self.device_train if self.training else self.device_eval
        x = x.to(device)
        return self.forward_action(x)


    def init_hidden(self, batch_size):
        device = self.device_train if self.training else self.device_eval
        for lists in self.Network_stack:
            for gru in lists:
                gru.init_hidden(batch_size, device)

    def train_enforce_constraints(self):
        Lip_lin.train_enforce_constraints()
        self.fc2.train_enforce_constraints()

if __name__ == '__main__':
    N_Agents = 2
    in_dim = 2
    batch_size = 32
    n_samples = 1024
    timesteps = 100
    lam = 0.5
    control_energy_reg = 1e-6 ### regularization on maximum control energy
    k_max = 64
    x = torch.randn(batch_size, N_Agents, in_dim)
    model = KAN_RNN_Layer(N_Agents = N_Agents, in_dim = in_dim, hidden = 256, depth = 2, n_timesteps = timesteps,sys_param_lam= lam, network_type = 'uni', device = 'cuda')
    model.train()
    model.init_hidden(batch_size)
    print(model(x)[0].shape, model(x)[1].shape, model(x)[1].device)
    model.eval()
    model.init_hidden(batch_size)
    print(model(x)[0].shape, model(x)[1].shape, model(x)[1].device)

    model = KAN_RNN_Layer(N_Agents = N_Agents, in_dim = in_dim, hidden = 256, depth = 2, n_timesteps = timesteps,sys_param_lam= lam, network_type = 'multi', device = 'cuda')
    model.train()
    model.init_hidden(batch_size)
    print(model(x)[0].shape, model(x)[1].shape, model(x)[1].device)
    model.eval()
    model.init_hidden(batch_size)
    print(model(x)[0].shape, model(x)[1].shape, model(x)[1].device)

