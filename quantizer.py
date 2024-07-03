import torch
from torch import nn
from torch import autograd

scales= []

class RoundSTE(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class RangeTracker(nn.Module):
    
    def __init__(self, momentum=0.1):
        super().__init__()
        self.min_val= torch.tensor(0) 
        self.max_val= torch.tensor(0)
        self.momentum = momentum
        self.max_vals = []
        self.min_vals = []
    
    @torch.no_grad()
    def forward(self, x):
        min_val = torch.min(x)
        max_val = torch.max(x)
        # print(max_val, min_val)
        self.max_vals.append(max_val.item())
        self.min_vals.append(min_val.item())
        self.update_range(min_val, max_val)
    
    def update_range(self, min_val, max_val):
        # ema를 사용하여 min_val, max_val을 업데이트
        if self.min_val == self.max_val :
            # 첫번째 값이 들어올 때
            self.min_val = min_val
            self.max_val = max_val
            return
        self.min_val = self.min_val * (1 - self.momentum) + min_val * self.momentum if self.min_val is not None else min_val
        self.max_val = self.max_val * (1 - self.momentum) + max_val * self.momentum if self.max_val is not None else max_val


class Quantizer(nn.Module):
    
    def __init__(self, num_bits, range_tracker=RangeTracker()):
        super().__init__()
        self.num_bits = num_bits
        self.range_tracker = range_tracker
        self.scale = None
        self.scales = []
        # self.zero_point = None
        self.min_val = torch.tensor(0)
        self.max_val = torch.tensor((1 << self.num_bits) - 1)
    
    def round(self, x):
        # round 후에는 gradient가 0이 되므로, STE를 사용하여 gradient를 전달
        return RoundSTE.apply(x)
    
    def clamp(self, x):
        return torch.clamp(x, self.range_tracker.min_val, self.range_tracker.max_val)
    
    def dequantize(self, x):
        return x * self.scale + self.range_tracker.min_val
    
    def quantize(self, x):
        if self.scale is None:
            self.scale = (self.range_tracker.max_val - self.range_tracker.min_val) / (self.max_val - self.min_val - 1)
        if self.scale == 0:
            print ('scale is 0', self.range_tracker.max_val, self.range_tracker.min_val)
            assert False
        return self.round((self.clamp(x) - self.range_tracker.min_val) / self.scale)
    
    def update_params(self):
        quantized_range = self.max_val - self.min_val - 1
        float_range = self.range_tracker.max_val - self.range_tracker.min_val
        
        # int only inference에 사용될 값인 scale과 zero_point를 업데이트
        self.scale = float_range / quantized_range
        self.scales.append(self.scale)
        # self.zero_point = self.quantize(torch.tensor(0))
        
    def forward(self, x):
        self.range_tracker(x)
        self.update_params()
        before = x
        x = self.quantize(x)
        x = self.dequantize(x)
        after = x
          
        return x