import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class depth_seperate_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(depth_seperate_conv, self).__init__()
        self.deep_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        if kernel_size == 1:
            self.downsample = nn.AvgPool2d(2,2)
        else:
            self.downsample = None
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        output = self.deep_conv(x)
        if self.downsample is not None:
            output = self.downsample(output)
        output = self.point_conv(output)
        return output


class ds_basic_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, time_emb_dim=None):
        super(ds_basic_conv, self).__init__()
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim,out_channels*2)
        )
        self.ds_conv = depth_seperate_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x, scale_shift):
        output = self.ds_conv(x)
        output = self.batchnorm(output)
        if scale_shift is not None:
            scale_shift = self.time_emb(scale_shift)
            scale_shift = rearrange(scale_shift, 'b c -> b c 1 1')
            scale, shift = scale_shift.chunk(2, dim=1)
            output = output * (scale + 1) + shift
        output = self.activation(output)
        return output
