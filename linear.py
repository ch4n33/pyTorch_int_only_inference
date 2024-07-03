import torch
from torch import nn
from quantizer import *
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, activation_quantizer=None, weight_quantizer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * math.sqrt(2 / in_features)) 
        # applied He initialization
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.activation_quantizer = activation_quantizer or Quantizer(8, RangeTracker())
        self.weight_quantizer = weight_quantizer or Quantizer(8, RangeTracker())
        # bias는 32bit precision으로, quantization 할 필요가 없음
        self.QAT = False
        self.max_vals = []

    def forward(self, x):
        if self.QAT:
            x = self.activation_quantizer(x)
            weight = self.weight_quantizer(self.weight)
            self.max_vals.append(torch.max(self.weight.flatten()).item())
        else:
            weight = self.weight
        # print (x.shape, weight.shape) : torch.Size([64, 3072]) torch.Size([512, 3072])
        return torch.matmul(x, weight.t()) + self.bias
    
    def enable_quantization(self):
        self.QAT = True
    
    def disable_quantization(self):
        self.QAT = False