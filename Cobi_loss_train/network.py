
import torch.nn as nn
from torch.nn import init
from base_networks import *


class Net(torch.nn.Module):
    def __init__(self, num_channels, scale_factor, d, s, m):
        super(Net, self).__init__()

        # Feature extraction
        self.first_part = ConvBlock(num_channels, d, 5, 1, 0, activation='prelu', norm=None)

        self.layers = []
        # Shrinking
        self.layers.append(ConvBlock(d, s, 1, 1, 0, activation='prelu', norm=None))
        # Non-linear Mapping
        for _ in range(m):
            self.layers.append(ConvBlock(s, s, 3, 1, 1, activation='prelu', norm=None))
        # self.layers.append(nn.PReLU())
        # Expanding
        self.layers.append(ConvBlock(s, d, 1, 1, 0, activation='prelu', norm=None))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.last_part = nn.ConvTranspose2d(d, num_channels, 9, scale_factor, 3, output_padding=1)



    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            # utils.weights_init_normal(m, mean=mean, std=std)
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()
    def weights_init_kaiming(self, scale=1):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
    # def weight_init_fromMAT_s(self, BasicInf_conv='./pretrained_model/tcl_x4_dark_words.mat'):
    #     print 'model.weight_init_fromMAT_s'
    #     basic = scipy.io.loadmat(BasicInf_conv)   #The data should be normalized firstly. GJW
    #
    #     biases1 = numpy.array(basic['biases1'])
    #     biases2 = numpy.array(basic['biases2'])
    #     biases3 = numpy.array(basic['biases3'])
    #     biases4 = numpy.array(basic['biases4'])
    #     biases5 = numpy.array(basic['biases5'])
    #
    #     conv1 = numpy.array(basic['conv1'])
    #     conv2 = numpy.array(basic['conv2'])
    #     conv3 = numpy.array(basic['conv3'])
    #     conv4 = numpy.array(basic['conv4'])
    #     conv5 = numpy.array(basic['conv5'])
    #     prelu_conv = numpy.array(basic['prelu_conv'])
    #     Conv_Num = 0
    #
    #         ### prelu ###
    #     self.first_part.act.weight.data = torch.from_numpy(prelu_conv[0])
    #     # conv = numpy.array(basic['conv1'])
    #     self.first_part._modules['conv'].weight.data = torch.from_numpy(conv1)
    #     self.first_part._modules['conv'].bias.data = torch.from_numpy(biases1[0,:])
    #
    #
    #     self.mid_part._modules['0'].conv._parameters['weight'].data = torch.from_numpy(conv2)[:,:,numpy.newaxis,numpy.newaxis]
    #     self.mid_part._modules['1'].conv._parameters['weight'].data = torch.from_numpy(conv3)
    #     self.mid_part._modules['2'].conv._parameters['weight'].data = torch.from_numpy(conv4)[:,:,numpy.newaxis,numpy.newaxis]
    #     self.mid_part._modules['0'].conv._parameters['bias'].data = torch.from_numpy(biases2)[0,:]
    #     self.mid_part._modules['1'].conv._parameters['bias'].data = torch.from_numpy(biases3)[0,:]
    #     self.mid_part._modules['2'].conv._parameters['bias'].data = torch.from_numpy(biases4)[0,:]
    #
    #     self.mid_part._modules['0'].act._parameters['weight'].data = torch.from_numpy(prelu_conv[1]).float()
    #     self.mid_part._modules['1'].act._parameters['weight'].data = torch.from_numpy(prelu_conv[2]).float()
    #     self.mid_part._modules['2'].act._parameters['weight'].data = torch.from_numpy(prelu_conv[3]).float()
    #
    #
    #     self.last_part.weight.data = torch.from_numpy(conv5)
    #     self.last_part.bias.data = torch.from_numpy(biases5)[0]
    #     #
    #     # for m in self.last_part.modules():
    #     #     if isinstance(m, nn.ConvTranspose2d):
    #     #         m.weight.data.normal_(0.0, 0.0001)
    #     #         if m.bias is not None:
    #     #             m.bias.data.zero_()







class Net_new3(torch.nn.Module):
    def __init__(self, num_channels, scale_factor, d, s, m):
        super(Net_new3, self).__init__()

        # Feature extraction
        self.first_part = ConvBlock(num_channels, 64, 5, 1, 0, activation='prelu', norm=None)

        self.layers = []
        # Shrinking
        self.layers.append(ConvBlock(64, 5, 1, 1, 0, activation='prelu', norm=None))
        # Non-linear Mapping
        for _ in range(2):
            self.layers.append(ConvBlock(5, 5, 3, 1, 1, activation='prelu', norm=None))
        # self.layers.append(nn.PReLU())
        # Expanding
        self.layers.append(ConvBlock(5, 64, 1, 1, 0, activation='prelu', norm=None))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.last_part1 = nn.ConvTranspose2d(64, 3, 9, 3, 3, output_padding=1)
        self.last_part2 = nn.ConvTranspose2d(3, num_channels, 9, 3, 3, output_padding=1)

    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part1(out)
        out = self.last_part2(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            # utils.weights_init_normal(m, mean=mean, std=std)
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()

    def weights_init_kaiming(self, scale=1):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()




### for scale 4X
class Net_new4(torch.nn.Module):
    def __init__(self, num_channels, scale_factor, d, s, m):
        super(Net_new4, self).__init__()

        # Feature extraction
        self.first_part = ConvBlock(num_channels, d, 5, 1, 0, activation='prelu', norm=None)

        self.layers = []
        # Shrinking
        self.layers.append(ConvBlock(d, s, 1, 1, 0, activation='prelu', norm=None))
        # Non-linear Mapping
        for _ in range(m):
            self.layers.append(ConvBlock(s, s, 3, 1, 1, activation='prelu', norm=None))
        # self.layers.append(nn.PReLU())
        # Expanding
        self.layers.append(ConvBlock(s, d, 1, 1, 0, activation='prelu', norm=None))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.las_part1 = nn.ConvTranspose2d(d, num_channels, 9, scale_factor, 3, output_padding=1)

    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.las_part1(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            # utils.weights_init_normal(m, mean=mean, std=std)
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()

    def weights_init_kaiming(self, scale=1):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()


    def weights_init_orthogonal(m):
        classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d):
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()

import math
from torch import nn


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
