import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def init_weights(modules):
    pass
   
class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale=4, group=1):
        super(UpsampleBlock, self).__init__()
        self.up =  _UpsampleBlock(n_channels, scale=scale, group=group)

    def forward(self, x):
	
        return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale,group=1):			 
        super(_UpsampleBlock, self).__init__()
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++(1)+++++++")
                #modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            #modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out
