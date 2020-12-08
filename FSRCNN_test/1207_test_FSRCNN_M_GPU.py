'''
This function is for test.
Given the model strcuture;
given the opretrained model;
and the test image file.De
You need to change the --test_dataset according to your case
And the result is the upscaled 4 images.
'''

is_train = False;
scale_factor = 4
Model_index = 1  
pretrained_model = 'lr_0.0001_x4_FSRCNN_epoch_24.pkl'
#pretrained_model = 'fsrcnn_x4.pth'

print ('scale_factor = %d' %scale_factor)


import torch
import os, argparse
import numpy as np
from PIL import Image 
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import *
import scipy.io
import numpy
import scipy.misc
import time
import torch._utils
from FSRCNN_network import FSRCNN
from os import listdir
from os.path import join
from PIL import Image
import random
import glob
import imageio

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

"""parsing and configuration"""
def parse_args():
    desc = "PyTorch implementation of S R collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--img_save_dir', type=str, default='1208_FSRCNN_test_epoch_24_Cobi_loss', help='Directory name to save validation pictures')
    parser.add_argument('--pretrained_model', type=str, default= pretrained_model, help='pretrained model')
    parser.add_argument('--test_dataset', type = str, default = 'lj_test_images')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

    assert args.test_dataset, 'Error: test_dataset path is not exist..'
    assert args.pretrained_model, 'Error: pretrained_model path is not exist..'


    return args
    
""" main """
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    model = FSRCNN(scale_factor=4, num_channels=1,  d=56, s=12, m=4)
    print("model")
    a = model.state_dict()
    print(a.keys())
    # print(model.state_dict())
    state_dict = torch.load(pretrained_model, map_location = torch.device('cpu'))
    print("state_dict")
    #print(state_dict.keys())

    # original saved file with DataParallel
    # create new OrderedDict that does not contain `module.`
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     print(k)
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    #     # load params
    # model.load_state_dict(new_state_dict)   

    # print(new_state_dict.keys())
    # model.load_state_dict(state_dict)

    # print(model)

    pretrained_state_dict = state_dict
    pretrained_state_dict_keys = list(pretrained_state_dict.keys())
    print(pretrained_state_dict.keys())
    net_model_state_dict = model.state_dict()
    #print(net_model_state_dict)
    net_model_state_dict_keys = list(net_model_state_dict.keys())
    #print("pretrain keys: ", len(pretrained_state_dict_keys))
    #print("netmodel keys: ", len(net_model_state_dict_keys))

    weight_load = {}
    for k in range(len(pretrained_state_dict_keys)):
         if pretrained_state_dict[pretrained_state_dict_keys[k]].shape == net_model_state_dict[net_model_state_dict_keys[k]].shape:
             weight_load[net_model_state_dict_keys[k]] = pretrained_state_dict[pretrained_state_dict_keys[k]]
             #print('init model', net_model_state_dict_keys[k], 'from pretrained', pretrained_state_dict_keys[k])
         else:
            break
        
    net_model_state_dict.update(weight_load)
    model.load_state_dict(net_model_state_dict)

    print(model)

    image_dir = args.test_dataset
    image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
    file_num = len(image_filenames)
    for idx in range(file_num):
        img_ycbcr = Image.open(image_filenames[idx]).convert('YCbCr') 
        img_y, img_cb, img_cr = img_ycbcr.split()
        #print(img_y.size, img_cb.size, img_cr.size)
    
        input_x_t = torch.from_numpy(numpy.zeros((1, 1, np.array(img_y).shape[0], np.array(img_y).shape[1]), dtype='f')) 
        
        #temp = torch.from_numpy(np.array(img_y))
        input_x_t[0, 0, :, :] = torch.from_numpy(np.array(img_y)/255)
        #print(input_x_t.size())
        #print(input_x_t.size()[2])
        #print(input_x_t.size()[3])
        recon_y = model(input_x_t).detach()
        temp_y = recon_y[0,0,:,:].numpy() * 255
        out_y = Image.fromarray(np.uint8(temp_y.clip(0,255)), mode="L")

        #out_y.show()
        ### use ToPILImage()
        #temp_y = torch.tensor(recon_y.squeeze(0).squeeze(0),dtype = torch.uint8)
        #out_y = transforms.ToPILImage()(temp_y)
        out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
        #out_cb.show()
        out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
        #out_cr.show()
        result = Image.merge('YCbCr',[out_y, out_cb, out_cr]).convert('RGB')
        #result.show()
        image_dir = args.img_save_dir

        if not os.path.exists(image_dir):
            os.mkdir(image_dir, mode=0o777)

    
        save_path = join(image_dir, str(idx)+'_SR' + '.png')
        imageio.imsave(save_path, result)
        #scipy.misc.imsave(save_path, result)
        idx += 1
        #print(idx)

    # save entire model
    

    #model_name_save = '1024_trainning_model_lr_0.0001_x4_Net_new4_epoch_500.pth'
    #torch.save(model, model_name_save)

    # from network import Net as net
    # model = net(num_channels=1, scale_factor=4, d=32, s=5, m=1)
    # model.load_state_dict(torch.load(args.pretrained_model))

    # model_name_save = './1104_trained_1480.onnx'
    # x=torch.randn(1,1,125,125,requires_grad=False).type(torch.float)
    # torch_out = model(x)
    # torch.onnx.export(model,x,model_name_save,export_params=True)

if __name__ == '__main__':

    main()

