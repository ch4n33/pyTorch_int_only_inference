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
        self.quantization = False

    def forward(self, x):
        if self.quantization:
            x = self.activation_quantizer(x)
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight
        return torch.matmul(x, weight.t()) + self.bias
