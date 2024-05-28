import torch
from torch import nn

class NormalizedMultiplication(nn.Module):
    # this normalization should be computed offline, not when infering. this is just for demonstration
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, M, product):
        if (M <= 0 or M >= 1):
            raise ValueError("M should be in [0, 1]")
        n = 0
        m = torch.tensor(M, dtype=torch.float32)
        while (m < 0.5):
            n += 1
            m *= 2
        # m * 2^-n is M
        m = torch.round(2**31 * m)
        m = torch.tensor(m, dtype=torch.int32)
        m >>= n
        
        return m * product
    

class IntArithmeticOnlyMult(nn.Module):
    def __init__(self):
        super().__init__()
        self.mult = NormalizedMultiplication()
    
    def forward(self, a, w, b, zero_a, zero_w, scale_a, scale_w):
        '''
        a: input tensor : torch.int8
        w: weight tensor: torch.int8
        b: bias tensor  : torch.int32
        zero_a: zero point of a: torch.int8
        zero_w: zero point of w: torch.int8
        scale_a: scale of a: torch.float32
        scale_w: scale of w: torch.float32
        '''
        a_16 = a.to(torch.int16)
        w_16 = w.to(torch.int16)

        row_sum = torch.sum(a_16, dim=1)
        col_sum = torch.sum(w_16, dim=0)
        
        hidden_dim = w_16.shape[0]
        
        product = (a_16 * w_16).to(torch.int32) \
                + hidden_dim * zero_a * zero_w \
                - zero_a * col_sum \
                - zero_w * row_sum
        
        zero_out = wtf?
        scale_out = wtf?
        
        M = scale_a * scale_w / scale_out
        
        product = zero_out + self.mult(M, product) \
                + b
        product = product.to(torch.int8) # is it correct?
        
        return product # did not passed activation function
        