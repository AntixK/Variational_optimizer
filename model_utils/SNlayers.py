import torch
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn
from spectral_norm import spectral_norm

class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.u = None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda())

    def forward(self, input):
        #print("renorm:",self.renorm.data)
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = spectral_norm(w_mat, self.u)
        self.u = _u
        self.weight.data = self.renorm.data * self.weight.data / sigma
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SNLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.u = None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda())
    def forward(self, input):
        w_mat = self.weight
        sigma, _u = spectral_norm(w_mat, self.u)
        self.u = _u
        self.weight.data = (self.weight.data / sigma) * self.renorm.data
        return F.linear(input, self.weight, self.bias)