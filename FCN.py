import math
import time

import torch as th
from torch import nn
from torch.nn import functional as F



class CBL_block(nn.Module):
    def __init__(self, in_channels=1, out_channels=30, kernel_size=55 ):
        super(CBL_block, self).__init__()
        #Convolution1D(30, 55, border_mode='same', input_shape=(None,1))

        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.bn1d = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):

        x = self.conv1d(x)

        x = self.bn1d(x)

        x = self.activation(x)

        return x
    
class FCN_denoise(nn.Module):
    def __init__(self, in_channels=1, out_channels=30, kernel_size=55 ):
        super(FCN_denoise, self).__init__()
        #Convolution1D(30, 55, border_mode='same', input_shape=(None,1))

        self.b1 = CBL_block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.b2 = CBL_block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.b3 = CBL_block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.b4 = CBL_block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.b5 = CBL_block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.b6 = CBL_block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.b7 = CBL_block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv1d = nn.Conv1d(in_channels=out_channels, out_channels=1, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.activation = nn.Tanh()
#         self.b3 = CBL_block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.b7(x)
        x = self.conv1d(x)

        x = self.activation(x)

        return x

def fcn_denoise(training=True, **kwargs):
    model=FCN_denoise()
    return model