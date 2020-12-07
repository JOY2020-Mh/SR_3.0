import torch
import torch.nn as nn
# from ._backbone import vgg19_extractor
try:
    import srlosses._backbone as cnn_backbone
except:
    import _backbone as cnn_backbone

class Perceptual_Loss(nn.Module):

    def __init__(self, backbone='vgg', pool='max'):
        super(Perceptual_Loss, self).__init__()
        self.compute_l1_loss = torch.nn.L1Loss()
        self.feature_extractor = getattr(cnn_backbone, backbone + '_extractor')(pool=pool)

    def forward(self, output, target, 
                features=['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2'], 
                withl1=False):
        t_feature=self.feature_extractor(target)
        o_feature=self.feature_extractor(output)
        loss_sum = 0
        if withl1:
            loss_sum += self.compute_l1_loss(t_feature['input'],o_feature['input'])
        if "conv1_2" in features:
            loss_sum += self.compute_l1_loss(t_feature['conv1_2'],o_feature['conv1_2'])
        if "conv2_2" in features:
            loss_sum += self.compute_l1_loss(t_feature['conv2_2'],o_feature['conv2_2'])
        if "conv3_2" in features:
            loss_sum += self.compute_l1_loss(t_feature['conv3_2'],o_feature['conv3_2'])/3.7 #need to modify
        if "conv4_2" in features:
            loss_sum += self.compute_l1_loss(t_feature['conv4_2'],o_feature['conv4_2'])/5.6 #need to modify
        if "conv5_2" in features:
            loss_sum += self.compute_l1_loss(t_feature['conv5_2'],o_feature['conv5_2'])*10/1.5 #need to modify
        return loss_sum
