import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._CX_helper import extract_image_patches, CX_loss_helper
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
from easydict import EasyDict as edict
from enum import Enum
try:
    import srlosses._backbone as cnn_backbone
except:
    import _backbone as cnn_backbone

class Distance(Enum):
    L2 = 0
    DotProduct = 1

class CoBi_Loss(nn.Module):
    def __init__(self, w_spatial=0.1, backbone='vgg19', pool='max', norm=False):
        super(CoBi_Loss, self).__init__()
        self.feature_extractor = getattr(cnn_backbone, backbone + '_extractor')(pool=pool, norm=norm)
        self.CX = edict()
        self.CX.crop_quarters = False
        self.CX.max_sampling_1d_size = 63
        self.CX.feat_layers = {'conv1_2' : 1.0, 'conv2_2' : 1.0, 'conv3_2': 0.5}
        self.CX.Dist = Distance.DotProduct # Distance.L2 # Distance.DotProduct
        self.CX.nn_stretch_sigma = 0.5 #0.1
        self.CX.w_spatial = w_spatial

    def forward(self, output, target):
        t_feature=self.feature_extractor(target)
        o_feature=self.feature_extractor(output)
        CX_loss_list = []
        for layer, w in self.CX.feat_layers.items():
            CX_loss_i = CX_loss_helper(t_feature[layer], o_feature[layer], self.CX)
            CX_loss_list.append(torch.unsqueeze(w * CX_loss_i, dim=0))
        
        CX_loss = torch.sum(torch.cat(CX_loss_list))
        return 5+CX_loss

class Patch_CoBi_Loss(nn.Module):
    def __init__(self, patch_sz=5, rates=1, w_spatial=0.1, backbone='vgg19', pool='max'):
        super(Patch_CoBi_Loss, self).__init__()
        # self.feature_extractor = getattr(cnn_backbone, backbone + '_extractor')(pool=pool)
        self.CX = edict()
        self.CX.crop_quarters = False
        self.CX.max_sampling_1d_size = 63
        self.CX.Dist = Distance.L2 # Distance.L2 # Distance.DotProduct
        self.CX.nn_stretch_sigma = 0.5 #0.1
        self.CX.w_spatial = w_spatial
        self.patch_sz = patch_sz
        self.rates = rates

    def forward(self, output, target):
        assert (type(output) is torch.Tensor)
        assert (type(target) is torch.Tensor)
        assert (torch.max(output) <=1 & torch.min(output) >=0) # only work for input ~ [0,1]
        assert (torch.max(target) <=1 & torch.min(target) >=0) # only work for input ~ [0,1]
        if output.shape[1] != 3:
            output = output.repeat(1, 3, 1, 1) # also work for gray scale
        if target.shape[1] != 3:
            target = target.repeat(1, 3, 1, 1) # also work for gray scale

        # to have the same scale as the VGG features
        target_patch = extract_image_patches(
            target, kernel=self.patch_sz, 
            stride=1,
            dilation=self.rates
        )

        output_patch = extract_image_patches(
            output, kernel=self.patch_sz, 
            stride=1,
            dilation=self.rates
        )
        #TODO cpu and cuda version
        CX_loss_i = CX_loss_helper(output_patch.cuda(), target_patch.cuda(), CX_config=self.CX)
        CX_loss = torch.sum(CX_loss_i)
        return CX_loss
