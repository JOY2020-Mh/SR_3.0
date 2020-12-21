import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import math
from math import exp



#############   gradient loss  #############
def compute_gradient(img):
    gradx=img[...,1:,:]-img[...,:-1,:]
    grady=img[...,1:]-img[...,:-1]
    return gradx,grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target) 
        
        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


#############   total variation loss  #############
class tv_loss(nn.Module):
    def __init__(self):
        super(tv_loss, self).__init__()

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self, img):
        """
        Compute total variation loss.
        Inputs:
        - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
        Returns:
        - loss: PyTorch Variable holding a scalar giving the total variation loss
          for img weighted by tv_weight.
        """
        batch_size = img.size()[0]
        h_x = img.size()[2]
        w_x = img.size()[3]
        count_h = self._tensor_size(img[:,:,1:,:])
        count_w = self._tensor_size(img[:,:,:,1:])
        h_tv = torch.pow((img[:,:,1:,:]-img[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((img[:,:,:,1:]-img[:,:,:,:w_x-1]),2).sum()
        loss = 2*(h_tv/count_h+w_tv/count_w)/batch_size
        return loss


#############   ssim loss  #############
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


#############   color loss  #############
class color_loss(nn.Module):
    def __init__(self):
        super(color_loss, self).__init__()
        self.dim = 1
        self.eps = 1e-6

    def forward(self, X, Y):
        cos = nn.CosineSimilarity(dim=self.dim, eps=self.eps)
        cos_sim = cos(X,Y)
        loss = torch.sum(cos_sim,2)
        loss = torch.sum(loss,1)
        loss = torch.div(loss, (X.shape[2] * X.shape[3]))
        loss = 1 - loss
        loss = torch.mean(loss) 
        return loss




#############   vgg loss  #############
class Vgg19(nn.Module):
    def __init__(self, args, requires_grad=False):
        super(Vgg19, self).__init__()
        self.args = args
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
    def forward(self, X):
        out = []
        for i in range(self.args.vgg_indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i+1) in self.args.vgg_indices:
                out.append(X)
        return out

class VGGLoss(nn.Module):
    def __init__(self, args):
        super(VGGLoss, self).__init__()
        self.args = args
        self.vgg_loss_net = Vgg19(args)
        self.vgg_loss_net.eval()



    def _vgg_preprocess(self, batch):
        # normalize using imagenet mean and std
        mean = torch.zeros_like(batch)
        std = torch.zeros_like(batch)
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        
        batch = (batch - mean) / std
        
        return batch


    def forward(self, X, Y):

        X = self._vgg_preprocess(X)
        Y = self._vgg_preprocess(Y)

        feat_X = self.vgg_loss_net(X)
        feat_Y = self.vgg_loss_net(Y)

        vgg_loss = 0
        for j in range(len(feat_X)):
            vgg_loss += self.args.lambda_vgg[j] * F.l1_loss(feat_X[j], feat_Y[j])

        return vgg_loss
