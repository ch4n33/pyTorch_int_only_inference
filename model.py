from torch import nn
from linear import Linear

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=3):
        super().__init__()
        self.input_layer = Linear(input_dim, hidden_dim)
        self.module_list = nn.ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.output_layer = Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        for module in self.module_list:
            x = module(x)
        return self.output_layer(x)