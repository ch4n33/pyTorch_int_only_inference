from torch import nn
from linear import Linear

class MLP(nn.Module):
    def __init__(self, input_dim=32*32*3, hidden_dim=100, output_dim=10, hidden_layers=3, activation=nn.ReLU()):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_layer = Linear(input_dim, hidden_dim)
        self.module_list = nn.ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.output_layer = Linear(hidden_dim, output_dim)
        self.activation = activation
        
        self.quantized = False
        
    def forward(self, x):
        x = self.flatten(x)
        # print(x.shape) : torch.Size([64, 3072])
        x = self.input_layer(x)
        for module in self.module_list:
            x = module(x)
            x = self.activation(x)
        return self.output_layer(x)
    
    def get_scales(self):
        return [module.weight_quantizer.scales for module in self.module_list]
    
    def get_max_vals(self):
        return [module.max_vals for module in self.module_list]
    
    def get_means(self):
        return [module.means for module in self.module_list]
    
    def get_bias_means(self):
        return [module.bias_means for module in self.module_list]
    
    def get_rangetrackers_max(self):
        return [module.activation_quantizer.range_tracker.max_vals for module in self.module_list]
    
    def get_rangetrackers_min(self):
        return [module.activation_quantizer.range_tracker.min_vals for module in self.module_list]