import torch
from torch import nn
from quantizer import *


class Linear(nn.Module):
    def __init__(self, in_features, out_features, activation_quantizer=None, weight_quantizer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.activation_quantizer = activation_quantizer or Quantizer(8, RangeTracker())
        self.weight_quantizer = weight_quantizer or Quantizer(8, RangeTracker())
        # bias는 32bit precision으로, quantization 할 필요가 없음
        self.QAT = False
        self.quantized = False

    def forward(self, x):
        if self.quantized:
            # ?
            return # quantized w, x을 
        if (torch.isnan(x).any()):
            print('x didnt have nan')
            assert False
        if (torch.isnan(self.weight).any()):
            print('weight didnt have nan')
            assert False
        if (torch.isnan(self.bias).any()):
            print('bias didnt have nan')
            assert False
        if self.QAT:
            x = self.activation_quantizer(x)
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight
        if (torch.isnan(x).any()):
            print('x has nan')
            assert False
        if (torch.isnan(weight).any()):
            print('weight has nan')
            assert False
        if (torch.isnan(self.bias).any()):
            print('bias has nan')
            assert False
        
        return torch.matmul(x, weight.t()) + self.bias
    
    def enable_quantization(self):
        self.QAT = True
    
    def disable_quantization(self):
        self.QAT = False