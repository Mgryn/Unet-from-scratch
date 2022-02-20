
import torch
import torch.nn as nn
from modules import *
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()        
        self.down_1 = Down(1, 64)
        self.down_2 = Down(64, 128)
        self.down_3 = Down(128, 256)
        self.down_4 = Down(256, 512)
        self.bottom = Convolutions(512, 1024)
        self.up_1 = Up(1024, 512)
        self.up_2 = Up(512, 256)
        self.up_3 = Up(256, 128)
        self.up_4 = Up(128, 64)
        self.last = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        x1, y1 = self.down_1(x)
        x2, y2 = self.down_2(y1)
        x3, y3 = self.down_3(y2)
        x4, y4 = self.down_4(y3)
        u0 = self.bottom(y4)
        u1 = self.up_1(u0, x4)
        u2 = self.up_2(u1, x3)
        u3 = self.up_3(u2, x2)
        u4 = self.up_4(u3, x1)
        out = self.last(u4)
        return out

ten = torch.randn(1, 1, 572, 572)
# m = nn.ConvTranspose2d(20, 40, (2,2))
# res = m(ten)
# print(res.shape)

u = Unet()
res = u.forward(ten)
print(res.shape)
# print(u)

