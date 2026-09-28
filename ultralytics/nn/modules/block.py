# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .conv import Conv

__all__ = ['DFL', 'SPPF', 'C2f', 'Bottleneck', 'ScaleAwareProgressivePConv', 'ProgressiveFasterBlock', 'C2f_ScaleAwarePConv', 'BPUScaleBlock', 'C2f_BPUScaleBlock']


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)










class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))






class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))














class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))






class BPUScaleBlock(nn.Module):
    """BPU-oriented pointwise-depthwise-pointwise compression bottleneck."""
    def __init__(self, c1, c2, alpha=0.25, shortcut=True, align=8):
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if int(align) < 1:
            raise ValueError("align must be positive")
        hidden = max(int(align), math.ceil(c2 * alpha / int(align)) * int(align))
        self.c1, self.c2, self.alpha, self.align, self.hidden = c1, c2, alpha, int(align), hidden
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.dw = Conv(hidden, hidden, 3, 1, g=hidden)
        self.cv2 = Conv(hidden, c2, 1, 1, act=False)
        self.use_residual = bool(shortcut and c1 == c2)

    def forward(self, x):
        y = self.cv2(self.dw(self.cv1(x)))
        return x + y if self.use_residual else y


class C2f_BPUScaleBlock(C2f):
    """C2f whose internal bottlenecks use BPUScaleBlock."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, alpha=0.25, align=8):
        if g != 1:
            raise ValueError('g must be 1 for this experiment')
        super().__init__(c1, c2, n, shortcut, g, e)
        self.alpha, self.align = alpha, int(align)
        self.m = nn.ModuleList(BPUScaleBlock(self.c, self.c, alpha, shortcut, align) for _ in range(n))

    def forward(self, x):
        return self.forward_split(x)

class ScaleAwareProgressivePConv(nn.Module):
    def __init__(self, dim, scale_type='medium'):
        super().__init__()
        if scale_type not in ('light', 'medium', 'heavy'):
            raise ValueError(f'Invalid scale_type: {scale_type}')
        self.scale_type = scale_type
        self.dim_conv3 = dim // {'light': 8, 'medium': 4, 'heavy': 2}[scale_type]
        self.dim_untouched = dim - self.dim_conv3
        if self.dim_conv3 < 1:
            raise ValueError('partial channels must be >= 1')
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3,
                                      3, 1, 1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        a, b = x.split((self.dim_conv3, self.dim_untouched), dim=1)
        return self.bn(torch.cat((self.partial_conv3(a), b), dim=1))


class ProgressiveFasterBlock(nn.Module):
    def __init__(self, dim, scale_type='medium', shortcut=True):
        super().__init__()
        self.spatial_mixing = ScaleAwareProgressivePConv(dim, scale_type)
        self.scale_type = scale_type
        self.add = shortcut
        if scale_type == 'medium':
            self.mlp = nn.Sequential(Conv(dim, dim, 1, g=dim), Conv(dim, dim, 1))
        else:
            hidden = int(dim * (0.5 if scale_type == 'light' else 1.5))
            self.mlp = nn.Sequential(Conv(dim, hidden, 1),
                                     nn.Conv2d(hidden, dim, 1, bias=False))

    def forward(self, x):
        y = self.mlp(self.spatial_mixing(x))
        return x + y if self.add else y


class C2f_ScaleAwarePConv(C2f):
    # YAML args after c2: shortcut, g, e, scale_type
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 scale_type='medium'):
        if g != 1:
            raise ValueError('g must be 1 for this experiment')
        super().__init__(c1, c2, n, shortcut, g, e)
        self.scale_type = scale_type
        self.m = nn.ModuleList(ProgressiveFasterBlock(self.c, scale_type, shortcut)
                               for _ in range(n))

    def forward(self, x):
        return self.forward_split(x)
