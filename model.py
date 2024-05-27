from torch import nn
from linear import Linear

class MLP(nn.Module):
    def __init__(self, input_dim=32*32*3, hidden_dim=100, output_dim=10, hidden_layers=3, activation=None):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_layer = Linear(input_dim, hidden_dim)
        self.module_list = nn.ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.output_layer = Linear(hidden_dim, output_dim)
        self.activation = activation or nn.ReLU6()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        for module in self.module_list:
            x = module(x)
            x = self.activation(x)
        return self.output_layer(x)