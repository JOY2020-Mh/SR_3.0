import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from os import listdir
from os.path import join
import imageio

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1] #h,w,channel [:-1] beside the final element, such as channel 
    x = random.randint(0, w-size) #random number
    y = random.randint(0, h-size)

    hsize = size*scale
    hx, hy = x*scale, y*scale

    crop_lr = lr[y:y+size, x:x+size].copy() #low-resolution patch
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize].copy()#high-resolution patch

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1) #left and right move
        im2 = np.flipud(im2)

    if random.random() < 0.5: #up and down move
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3]) #rote
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()

        self.size = size
        h5f = h5py.File(path, "r")
        
        self.hr = [v[:] for v in h5f["HR"].values()]
        # perform multi-scale training
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
            
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]
        
        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        size = self.size
        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]
        #i is the number, _ is the context
        item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]
        
        return [(self.transform(hr), self.transform(lr)) for hr, lr in item] #tranform array (hr,lr) into tensor

    def __len__(self):
        return len(self.hr)
        

class TestDataset(data.Dataset):
    def __init__(self, image_dir, scale):
        super(TestDataset, self).__init__()

        # self.name  = dirname
        self.scale = scale
        self.lr = [join(image_dir, x) for x in sorted(listdir(image_dir))]
           
        #self.hr.sort()
        self.lr.sort()
        print(len(self.lr))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        #hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])
        '''
        #This is only used to test code, which makes me more know the code. 
        ss = np.asarray(hr) (644,1024,3)
        print ss.shape
        '''
        #hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        lr_w = lr.size[0]
        lr_h = lr.size[1]

        lr_crop_h = 125
        lr_crop_w = 125
        ### center crop
        rnd_h = (lr_h - lr_crop_h)//2
        rnd_w = (lr_w - lr_crop_w)//2
        lr = lr.crop((rnd_w, rnd_h, rnd_w + lr_crop_w, rnd_h + lr_crop_h))
        '''
        tss = np.asarray(hr) (644,1024,3)
        print tss.shape
        '''
        filename = self.lr[index].split("/")[-1]

        #new_LR_path = 'testing_crop_image/img_LR'+str(index)+'.png'

        #imageio.imsave(new_LR_path, lr)
        #keep the last string, which is after /. For example, a= 'aaa/asds.bmp', a.split("/")[-1] = asds 
        return self.transform(lr), filename

    def __len__(self):
        return len(self.hr)
