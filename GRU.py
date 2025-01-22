import torch
from torch import nn

class GRU(nn.Module):
    """
        Implements a Gated Recurrent Unit
        Inputs [Batch_size, N_Agents, in_dim]
        returns:
            [Num_timesteps, Batch_size, N_Agents, in_dim]

    """
    def __init__(self, in_dim, N_Agents ,hidden, n_timesteps, depth = 2, activation = nn.ReLU):
        super(GRU, self).__init__()
        #Problem Params
        self.in_dim = in_dim
        self.N_Agents = N_Agents
        self.n_timesteps = n_timesteps
        self.sys_param_lam = 0.5
        self.L = [1. for _ in range(in_dim)]
        ## Model Params
        self.device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu") #TODO implement device method as in KAN_RNN
        self.device_eval = torch.device("cpu")
        self.hidden = hidden
        self.depth = depth
        self.activation = activation()
        self.gru = nn.GRU(in_dim * N_Agents, self.hidden, num_layers = self.depth, batch_first=True)
        self.fc1 = nn.Linear(self.hidden, in_dim * N_Agents)
        self.initialize_weights() 

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'weight' in name:  # Linear layer weights
                torch.nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param) 
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

    def init_hidden(self, batch_size):
        device = self.device_train if self.training else self.device_eval
        self.hidden_state = (torch.randn(self.depth, batch_size, self.hidden) * 0.01).to(device) 
        ## I hope randn weights  elp in faster convergence

    def time_step(self, x):
        out, self.hidden_state = self.gru(x, self.hidden_state)
        return self.fc1(self.activation(out))

    def system_dynamics(self,u, x_prev):
        """
            x_prev = [Batch_size, N_Agents, in_dim]
        """
        x_new = torch.zeros_like(x_prev)
        thres = 0.
        x_state = (x_prev + self.sys_param_lam * u)
        for i in range(self.in_dim):
            dim_idx = torch.arange(i, x_state.shape[2], step=self.in_dim)
            x_new[:,:,dim_idx] = torch.clamp(x_state[:,:,dim_idx], min = 0, max = self.L[i])
            if self.training:
                self.penalty += torch.sum((torch.relu(-x_state - thres))**2) + torch.sum((torch.relu(x_state - self.L[i] - thres))**2)
        return x_new
        
    def to_device(self):
        device = self.device_train if self.training else self.device_eval
        self.to(device)

    def forward(self,x):
        self.penalty = 0
        device = self.device_train if self.training else self.device_eval
        x = x.to(device)
        outs = torch.zeros(self.n_timesteps, *x.shape, device = device)
        control_trajectory = torch.zeros(self.n_timesteps, *x.shape, device = device) ## Assume u of same shape as x!
        for i in range(self.n_timesteps):
            outs[i] = x
            u = self.time_step(x)
            x = self.system_dynamics(u, x)
            control_trajectory[i] = u
        return outs, control_trajectory





