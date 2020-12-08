import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import numpy as np

class vgg19_extractor(nn.Module):
    def __init__(self, pool='max', norm=False):
        super(vgg19_extractor, self).__init__()
        if torch.cuda.is_available():
            self.vgg_layers=vgg19(pretrained=True).cuda().features
        else:
            self.vgg_layers=vgg19(pretrained=True).features
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        assert pool in ('max', 'avg') # for the implementations in (https://github.com/ceciliavision/zoom-learn-zoom) is avgpooling
        self.pool = pool
        self.norm = norm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), requires_grad=False) #TODO need to make sure
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), requires_grad=False) #TODO need to make sure
    def forward(self, input):
        assert (type(input) is torch.Tensor)
        assert ((torch.max(input) <=1) & (torch.min(input) >=0)) # only work for input ~ [0,1]
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1) # also work for gray scale
        input = input.to(self.device)
        if self.norm:
            input = (input-self.mean.to(self.device)) / self.std.to(self.device) # normalize for imagenet pretrained model
        net = {}
        # Easy to extract the features of specific layer
        # with torch.no_grad():
        # net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,3,1,1)) #TODO why the input image is 0-255
        net['input']=input
        net['conv1_1'] = self.vgg_layers[0](net['input'].float()) #conv
        x = self.vgg_layers[1](net['conv1_1']) #relu 
        net['conv1_2'] = self.vgg_layers[2](x) #conv
        x = self.vgg_layers[3](net['conv1_2']) #relu
        if self.pool == 'max':
            net['pool1'] = self.vgg_layers[4](x) #pool
        else:
            net['pool1'] = nn.AvgPool2d(2,2,padding=0)(x) #pool
        net['conv2_1'] = self.vgg_layers[5](net['pool1']) #conv
        x = self.vgg_layers[6](net['conv2_1']) #relu
        net['conv2_2'] = self.vgg_layers[7](x) #conv
        x = self.vgg_layers[8](net['conv2_2']) #relu
        if self.pool == 'max':
            net['pool2'] = self.vgg_layers[9](x) #pool
        else:
            net['pool2'] = nn.AvgPool2d(2,2,padding=0)(x) #pool
        net['conv3_1'] = self.vgg_layers[10](net['pool2']) # conv
        x = self.vgg_layers[11](net['conv3_1']) #relu
        net['conv3_2'] = self.vgg_layers[12](x) #conv
        x = self.vgg_layers[13](net['conv3_2']) #relu
        net['conv3_3'] = self.vgg_layers[14](x) #conv
        x = self.vgg_layers[15](net['conv3_3']) #relu
        net['conv3_4'] = self.vgg_layers[16](x) #conv
        x = self.vgg_layers[17](net['conv3_4']) #relu
        if self.pool == 'max':
            net['pool3'] = self.vgg_layers[18](x) #pool
        else:
            net['pool3'] = nn.AvgPool2d(2,2,padding=0)(x) #pool
        net['conv4_1'] = self.vgg_layers[19](net['pool3']) # conv
        x = self.vgg_layers[20](net['conv4_1']) #relu
        net['conv4_2'] = self.vgg_layers[21](x) #conv
        x = self.vgg_layers[22](net['conv4_2']) #relu
        net['conv4_3'] = self.vgg_layers[23](x) #conv
        x = self.vgg_layers[24](net['conv4_3']) #relu
        net['conv4_4'] = self.vgg_layers[25](x) #conv
        x = self.vgg_layers[26](net['conv4_4']) #relu
        if self.pool == 'max':
            net['pool4'] = self.vgg_layers[27](x) #pool
        else:
            net['pool4'] = nn.AvgPool2d(2,2,padding=0)(x) #pool
        net['conv5_1'] = self.vgg_layers[28](net['pool4']) # conv
        x = self.vgg_layers[29](net['conv5_1']) #relu
        net['conv5_2'] = self.vgg_layers[30](x) #conv
        return net