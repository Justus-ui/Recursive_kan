import torch
import torch.nn as nn

class Lipschitz_GRU(nn.Module):
    """
        Defines a GRU with enforceable LIpschitz condition
    """
    def __init__(self, in_dim ,hidden, depth = 2, activation = nn.ReLU):
        super(Lipschitz_GRU, self).__init__()
        self.hidden = hidden
        self.depth = depth

        self.B = 2 ## Upper bound on product of weights norms
        self.lip_reg = 0.0005 ## learning rate on Lip regularisation !
        self.order = float('inf') ### order of L_j norm
        #self.activation = activation ##TODO when does this work and when not??
        self.activation = nn.ReLU()
        self.Norm_constraints = torch.rand(self.depth) * self.B
        self.gru = nn.GRU(in_dim, self.hidden, num_layers = self.depth, batch_first=True)
        
    def forward(self, x):
        out, self.hidden_state = self.gru(x, self.hidden_state)
        return out
    
    def init_hidden(self, batch_size):
        self.hidden_state = torch.zeros(self.depth, batch_size, self.hidden)
    
    def compute_constraint_gradient(self):
        self.grads_norm_constraints = []
        prod = torch.prod(self.Norm_constraints)
        for i in range(len(self.Norm_constraints)):
            grad = torch.sum(self.linear_layers[i].weight.grad * (self.linear_layers[i].weight.data / (torch.linalg.matrix_norm(self.linear_layers[i].weight.data, ord = self.order)))) + self.lip_reg * (prod / self.Norm_constraints[i])  * torch.exp(7 * ((prod) - self.B))
            self.grads_norm_constraints.append(grad)

    def upper_lipschitz_bound(self):
        return torch.prod(self.Norm_constraints)

    def update_norm_constraints(self):
        for i in range(len(self.Norm_constraints)):
            self.Norm_constraints[i] -= 0.001 * self.grads_norm_constraints[i]
    
    def project_on_norm_constraints(self):
        for i in range(len(self.Norm_constraints)):
            self.linear_layers[i].weight.data *= self.Norm_constraints[i] / torch.linalg.matrix_norm(self.linear_layers[i].weight.data, ord = self.order)
    
    def train_enforce_constraints(self):
        self.compute_constraint_gradient()
        self.update_norm_constraints()
        self.project_on_norm_constraints()



