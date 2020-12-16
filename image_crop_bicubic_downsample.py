import torch.utils.data as data
from torchvision.transforms import *
from os import listdir
from os.path import join
from PIL import Image
import random
import glob
from PIL import ImageFile
import imageio
import os
#ImageFile.LOAD_TRUNCATED_IMAGES = True

scale_factor = 4
crop_size = 256
GT_dir = '/home/miaohuan/Documents/dataset_SR/Visstyle_Dataset_x4/asf_off1/bicubic_LR'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

HR_save_dir = 'mixed_HR1'
if not os.path.exists(HR_save_dir):
        os.mkdir(HR_save_dir, mode=0o777)
LR_save_dir = 'mixed_LR1'
if not os.path.exists(LR_save_dir):
        os.mkdir(LR_save_dir, mode=0o777)


GT_image_filenames = []
for image_name in sorted(glob.glob(GT_dir+'/*')):
    GT_image_filenames.append(image_name)

print(GT_image_filenames)

for index in range(len(GT_image_filenames)):

        GT_img = Image.open(GT_image_filenames[index])
        print('++++GT_img.size()++++',GT_img.size)

        hr_h = GT_img.size[1]
        hr_w = GT_img.size[0]
        lr_h = hr_h//scale_factor
        lr_w = hr_w//scale_factor
        transform_lr = Resize((lr_h, lr_w), interpolation=Image.BILINEAR)
        hr_h = lr_h * scale_factor
        hr_w = lr_w * scale_factor
        if min(hr_h,hr_w) < 256:
            print(index)
            print(GT_image_filenames[index])
            continue

        transform_hr = Resize((hr_h, hr_w), interpolation=Image.BILINEAR)
        LR_img = transform_lr(GT_img)
        HR_img = transform_hr(GT_img)
        print('LR.size; HR.size', LR_img.size, HR_img.size)

        save_hr_path = join(HR_save_dir, str(index) + '.png')
        save_lr_path = join(LR_save_dir, str(index) + '.png')
        imageio.imsave(save_hr_path, HR_img)
        imageio.imsave(save_lr_path, LR_img)
