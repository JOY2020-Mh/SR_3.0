import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import skimage.measure as measure #tcw201904101622tcw
from torch.autograd import Variable
from dataset_1203 import TestDataset
from PIL import Image
import cv2 #201904111751tcwi
#from torchsummary import summary #tcw20190623
#from torchsummaryX import summary #tcw20190625
os.environ['CUDA_VISIBLE_DEVICES']='3'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = 'lesrcnn')
    parser.add_argument("--ckpt_path", type=str, default = '1207_Steven_4x_2019data_lesrcnn_x4_1939.pth')
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--sample_dir", type=str, default = './1207_Steven_4x_2019data_lesrcnn_x4_1939_SR_images')
    parser.add_argument("--test_data_dir", type=str, default="../lj_test_images")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()

def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

#tcw20190413043
def sample(net, device, dataset, cfg):
    scale = cfg.scale
    mean_psnr = 0
    mean_psnr1 = 0
    mean_psnr2 = 0
    mean_ssim = 0 #tcw20190413047
    for step, (lr, name) in enumerate(dataset):

        t1 = time.time()
        #print '--------'
        print(lr.size()) #e.g (3,512,512)
        lr = lr.unsqueeze(0).to(device)
        print(lr.size())
        #print lr.size() #(1,3,512,512)
        #b = net(lr, cfg.scale).detach()
        #print b.size()  #(1,3,1024,1024)
        sr = net(lr, cfg.scale).detach().squeeze(0) #detach() break the reversed transformation.
        print(sr.size()) #(3,1024,1024)
        lr = lr.squeeze(0)
        #print lr.size() #(3,512,512)
        t2 = time.time()
        #print model_name
        sr_dir = cfg.sample_dir
        if not os.path.exists(sr_dir): #201904072208 tcw
            os.makedirs(sr_dir, mode=0o777)
        
        sr_name = name.split('.jpg')[0] + '_SR.png'
        sr_im_path = os.path.join(sr_dir, sr_name) 
        save_image(sr, sr_im_path)
 

def main(cfg):
    module = importlib.import_module("model.{}".format(cfg.model))
    ''' 
    net = module.Net(multi_scale=False, 
                     group=cfg.group)
    '''
    net = module.Net(scale=cfg.scale, 
                     group=cfg.group)
    #print(net)
    '''
    #net = MyModel
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
	#print('' + str(list(i.size())))
	for j in i.size():
            l *= j
	    #print('' + str(l))
	    k = k + l
	print(''+ str(k))
    '''
    #print(json.dumps(vars(cfg), indent=4, sort_keys=True)) #print cfg information according order.
    state_dict = torch.load(cfg.ckpt_path, map_location = 'cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    #os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #0 is number of gpu, if this gpu1 is work, you can set it into 1 (device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
    net = net.to(device)
    #summary(net,[(3,240, 160),(3,1000, 2000)]) #tcw20190623
    #summary(net,[torch.zeros(1,3,240,160),2],2)

    dataset = TestDataset(cfg.test_data_dir, cfg.scale)
    sample(net, device, dataset, cfg)
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
