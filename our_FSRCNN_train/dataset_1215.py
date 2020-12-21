import torch.utils.data as data
from torchvision.transforms import *
from os import listdir
from os.path import join
from PIL import Image
import random
import glob
from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath, inYCbCr=False):
    if inYCbCr:
        img = Image.open(filepath).convert('YCbCr') 
    else:
        img = Image.open(filepath).convert('RGB')
    return img


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)


class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, GT_dir, LR_dir, is_gray=False, random_scale=False, crop_size=128, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=4, save_inImg=False, inYCbCr=True):
        super(TrainDatasetFromFolder, self).__init__()
        self.GT_image_filenames = []
        self.LR_image_filenames = []
        for image_name in sorted(glob.glob(GT_dir+'/*')):
            self.GT_image_filenames.append(image_name)
        for image_name in sorted(glob.glob(LR_dir+'/*')):
            self.LR_image_filenames.append(image_name)


        self.is_gray = is_gray
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.scale_factor = scale_factor
        self.save_inImg = save_inImg
        self.inYCbCr = inYCbCr

    def __getitem__(self, index):
        # load image
        # print self.GT_image_filenames[index]
# print self.LR_image_filenames[index]
        GT_img = load_img(self.GT_image_filenames[index], self.inYCbCr)
        LR_img = load_img(self.LR_image_filenames[index], self.inYCbCr)
        
        #print('++++GT_img.size()++++',GT_img.size)
        hr_h = GT_img.size[1]
        hr_w = GT_img.size[0]
        #resize LR image to the needed size
        lr_h = hr_h//self.scale_factor
        lr_w = hr_w//self.scale_factor
        #transform_lr = Resize((lr_w, lr_h), interpolation=Image.BILINEAR)
        #transform_hr = REsize((hr_h, hr_w), interpolation=Image.BILINEAR)
        #hr_h = lr_h * self.scale_factor
        #hr_w = lr_w * self.scale_factor
        #if min(hr_h, hr_w)<256:
        #    continue

        #transform_hr = Resize((hr_w, hr_h), interpolation=Image.BILINEAR)
        #LR_img = transform_lr(GT_img)
        #HR_img = transform_hr(GT_img)
        #print('LR.size; HR.size', LR_img.size, GT_img.size)
        # print self.GT_image_filenames[index]+ 'GT,' + self.LR_image_filenames[index] + 'LR'
        # determine valid HR image size with scale factor
        # self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_crop_w = self.crop_size
        hr_crop_h = self.crop_size
        #lr_h = hr_h//self.scale_factor
        #lr_w = hr_w//self.scale_factor
        #transform = Resize((lr_h, lr_w), interpolation=Image.BILINEAR)
        #LR_img = transform(LR_img)

        # determine LR crop image size
        lr_crop_w = hr_crop_w // self.scale_factor
        lr_crop_h = hr_crop_h // self.scale_factor

        # center crop
        #rnd_h = (lr_h - lr_crop_h)//2
        #rnd_w = (lr_w - lr_crop_w)//2
        #ramdom crop image
        rnd_h = random.randint(0, max(0, lr_h - lr_crop_h - 1))
        rnd_w = random.randint(0, max(0, lr_w - lr_crop_w - 1))
        img_LR = LR_img.crop((rnd_w, rnd_h, rnd_w + lr_crop_w, rnd_h + lr_crop_h))

        #rnd_h_HR = (hr_h- hr_crop_h)//2
        #rnd_w_HR = (hr_w- hr_crop_w)//2
        rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale_factor), int(rnd_w * self.scale_factor)
        img_HR = GT_img.crop((rnd_w_HR, rnd_h_HR, rnd_w_HR + hr_crop_w, rnd_h_HR + hr_crop_h))

        # if self.save_inImg == True:
        # img_HR.save('train_image/img_HR'+str(index)+'.png')
        # img_LR.save('train_imagetrain_image/img_LR'+str(index)+'.png')

        img_HR=ToTensor()(img_HR)
        img_LR=ToTensor()(img_LR)

        return {'img_LR': img_LR, 'img_HR': img_HR}
        ### croped img_HR and img_LR

    def __len__(self):
        return len(self.GT_image_filenames)


class TestDatasetFromFolder_LR(data.Dataset):
    def __init__(self, image_dir, inYCbCr=False, scale_factor=4, is_LR=False):
        super(TestDatasetFromFolder_LR, self).__init__()

        self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.inYCbCr = inYCbCr
        self.is_LR = is_LR
        self.scale_factor = scale_factor


    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index], self.inYCbCr)

        # original HR image size
        w = img.size[0]
        h = img.size[1]
        #fovzoom = 0
        #if fovzoom:
        #    img = img.crop((int((h - h//fovzoom) // 2), int((w - w//fovzoom) // 2), int((h + h//fovzoom) // 2), int((w + w//fovzoom) // 2)))
        lr_img = transforms.ToTensor()(img)
        # if self.is_LR == False:
        #     # determine lr_img LR image size, downscale it
        #     hr_crop_w = calculate_valid_crop_size(w, self.scale_factor)
        #     hr_crop_h = calculate_valid_crop_size(h, self.scale_factor)
        #     lr_crop_w = hr_crop_w // self.scale_factor
        #     lr_crop_h = hr_crop_h // self.scale_factor
        #     lr_transform = Compose([Resize((lr_crop_h, lr_crop_w), interpolation=Image.BICUBIC), ToTensor()])
        #     lr_img = lr_transform(img)
        # else:
        #     #keep the same size and return
        #     lr_img = transforms.ToTensor()(img)

        return lr_img


    def __len__(self):
        return len(self.image_filenames)

class TestDatasetFromFolder_GT(data.Dataset):
    def __init__(self, image_dir, inYCbCr=False, is_gray=False, scale_factor=4,  is_LR = False):
        super(TestDatasetFromFolder_GT, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.inYCbCr = inYCbCr
        self.is_gray = is_gray
        self.is_LR = is_LR
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index], self.inYCbCr)

        # cal the original HR image size or is HR
        #if self.is_LR == True:
        #    w = img.size[0]*self.scale_factor
        #    h = img.size[1]*self.scale_factor
        #else:
        w = img.size[0]
        h = img.size[1]

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            # img, _, _ = lr_img.split()

        # hr_img HR image
        hr_transform = Compose([ToTensor()])
        hr_img = hr_transform(img)

        return hr_img

    def __len__(self):
        return len(self.image_filenames)
