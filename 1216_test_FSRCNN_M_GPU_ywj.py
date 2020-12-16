
is_valid = True
is_train = False
scale_factor = 4
Model_index = 2   #0 for no loading pretrained model, 1 for loading pretrained model
#model_name = '4x/visidon_style/vistyle_day_v2.pkl'
model_name = 'Net'
print ('scale_factor = %d' %scale_factor)
#pretrained_model = '1118_visdon_HR_downsampling_ssim_Net_epoch_90.pkl'

from network import FSRCNN
import torch
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from os import listdir
from os.path import join
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
import imageio
import os


"""parsing and configuration"""


class Args():
    def __init__(self, pretrained_model):
        self.pretrained_model = pretrained_model
        self.test_image_dir = 'lj_test_images'
        self.scale = 4
        self.image_save_dir = '1216_mixed_public_l1_ssim_loss'


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # parse arguments
    pkl_file = '1216_Mixed_public_data_our_FSRCNN_l1_ssim_loss_lr_decay_save_model'
    for filename in sorted(os.listdir(pkl_file)):
        #print(filename)
        if os.path.splitext(filename)[1] == '.pkl':
            print(filename)

            save_img_name = 'epoch' + '_' + filename.split('.')[-2].split('_')[-1]
            print(save_img_name)
            pretrained_model = join(pkl_file, filename)
            print(pretrained_model)

            args = Args(pretrained_model)

            print(args.pretrained_model)
     
            ####load model from M GPU
            model = FSRCNN(scale_factor=4, num_channels=1, d=56, s=12, m=4)

            state_dict = torch.load(args.pretrained_model, map_location = 'cpu')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
                # load params
            model.load_state_dict(new_state_dict) 
            #model = TCL_SuperResolution(args)
            model.eval()
            ###
            image_dir = args.test_image_dir
            image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir))]
            ### save the testing dataset
            image_save_dir = args.image_save_dir
            if not os.path.exists(image_save_dir):
                os.mkdir(image_save_dir, mode=0o777)
            
            file_num = len(image_filenames)

            for idx in range(file_num):
                image = pil_image.open(image_filenames[idx]).convert('RGB')
                lr = image
                bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
                lr, _ = preprocess(lr, device)
                # hr, _ = preprocess(hr, device)
                _, ycbcr = preprocess(bicubic, device)

                with torch.no_grad():
                    preds = model(lr).clamp(0.0, 1.0)
    
                # psnr = calc_psnr(hr, preds)
                # print('PSNR: {:.2f}'.format(psnr))
    
                preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                out_Y = pil_image.fromarray(np.uint8(preds), mode='L')
                # save_path = 'Y_our_fsrcnn_DIV_Flickr_epoch_91' + str(idx) + '.png'
                # out_Y.save(save_path)

                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
                output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
                output = pil_image.fromarray(output)

                final_image_name = str(idx) '_' + 'fsrcnn' + '_' + save_img_name + '.png'
                print(final_image_name)
                save_path = join(image_save_dir, final_image_name)
                imageio.imsave(save_path, output)
                # output.save(image_filenames[idx].replace('.', '_fsrcnn_x{}.'.format(args.scale)))
    
if __name__ == '__main__':
    
    main()

