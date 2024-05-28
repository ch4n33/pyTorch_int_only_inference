from linear import *


class QuantizationEnabler():

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for module in self.model.modules():
            if isinstance(module, Linear):
                module.enable_quantization()
        print('Quantization enabled')

    def __exit__(self, exc_type, exc_value, traceback):
        for module in self.model.modules():
            if isinstance(module, Linear):
                module.disable_quantization()
        print('Quantization disabled')


class QuantizedReLU(nn.Module):
    def forward(self, x, zero_point):
        return torch.clamp(x, zero_point, 127)