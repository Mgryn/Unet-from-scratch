import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class Convolutions(nn.Module):
    "Duble convolutions withe ReLU activation"
    def __init__(self, in_channels, out_channels):
        super(Convolutions, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class Down(nn.Module):
    "Double convolution followed by max poolinx"
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.convolution = Convolutions(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x_c = self.convolution(x)
        x_mp = self.max_pool(x_c)
        return x_c, x_mp

class Up(nn.Module):
    "Upsamping followed by double convolution."
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv_up = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2)
        self.convolution = Convolutions(in_channels, out_channels)
    
    def forward(self, x, x_down):
        x_up = self.conv_up(x)
        shape_diff = (x_down.shape[2] - x_up.shape[2])
        x_down = F.pad(
            x_down,
            (
                -shape_diff // 2,
                -int(np.floor(shape_diff / 2)),
                -shape_diff // 2,
                -int(np.floor(shape_diff / 2))
                )
            )
        x_cat = torch.cat([x_down, x_up], dim=1)
        x_cat = self.convolution(x_cat)
        return x_cat
