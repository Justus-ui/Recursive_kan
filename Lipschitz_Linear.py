import torch
import torch.nn as nn

class Lipschitz_Linear(nn.Module):
    """
        NN Layer with enforceable Lipschitz constraint.
    """
    def __init__(self, h, activation = None):
        super(Lipschitz_Linear, self).__init__()
        self.B = 2 ## Upper bound on product of weights norms
        self.lip_reg = 0.0005 ## learning rate on Lip regularisation !
        self.order = float('inf') ### order of L_j norm
        h = h
        self.activation = activation
        layers = []
        self.linear_layers = []
        self.Norm_constraints = torch.rand(len(h) - 1) * self.B
        for layer in range(1,len(h)):
            linear = nn.Linear(h[layer -1], h[layer])
            layers.append(linear)
            #layers.append(nn.BatchNorm1d(h[layer], affine=True))
            self.linear_layers.append(linear)
            if activation is not None:
                layers.append(self.activation())
        self.univariate_nn = nn.Sequential(*layers)
        
    def forward(self,x):
        return self.univariate_nn(x)
    
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



