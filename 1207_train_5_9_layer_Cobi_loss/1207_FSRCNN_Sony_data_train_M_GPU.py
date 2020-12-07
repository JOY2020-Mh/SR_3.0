
is_train = True
scale_factor = 4
#Model_index = 2   #0 for no loading pretrained model, 1 for loading pretrained model
#model_name = '4x/visidon_style/vistyle_day_v2.pkl'
model_name = 'Net'
print ('scale_factor = %d' %scale_factor)
from TCL_SuperResolution_Model_FSRCNN_M_GPU import TCL_SuperResolution



import torch
import os, argparse


"""parsing and configuration"""

def parse_args():
    desc = "PyTorch implementation of S R collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--is_train', type=bool, default=is_train)
    parser.add_argument('--model_name', type=str, default='Net',
                        choices=['Net', 'Net_new4', 'Net_new3'], help='The type of model')
    parser.add_argument('--scale_factor', type=int, default=scale_factor, help='Size of scale factor')
    parser.add_argument('--Model_index', type=int, default=0)  ### 0 is model used until V1.6. Rest represent the update index.
    parser.add_argument('--loss_func', type=str, default='ssim') ##mse, ssim
    parser.add_argument('--GT_dir', type=str, default='./SR_train/HR')
    parser.add_argument('--LR_dir', type=str, default='./SR_train/LR')
    #parser.add_argument('--GT_dir', type=str, default='../data/T1/asf_off/ppt/HR')
    #parser.add_argument('--LR_dir', type=str, default='../data/T1/asf_off/ppt/LR')
    parser.add_argument('--inYCbCr', type=bool, default=False)
    parser.add_argument('--save_inImg', type=bool, default=False)

    parser.add_argument('--is_LR', type=bool, default=True)
    parser.add_argument('--test_GT', type=str, default= './SR_validation/DIV2K_validation_HR') #//'../data/Set5/HR')
    parser.add_argument('--test_dataset', type=str, default= './SR_validation/DIV_2K_validation_bicubic_4x_LR')#'/home/pancc/eclipse-workspace/TCLSuperRes/data/Btest4/burst_1frame/LR/LRless') # to be logically optimized.......
    parser.add_argument('--model_save_dir', type=str, default='./1029_9_layer_trainning_model', help='Directory name to save models')
    parser.add_argument('--img_save_dir', type=str, default='./experiments/1029_9layer', help='Directory name to save validation pictures')
    parser.add_argument('--pretrained_model', type=str, default='1024_trainning_modellr_0.0001_x4_Net_new4_epoch_500.pth', help='Directory name to pretrained model')

    parser.add_argument('--crop_size', type=int, default=500, help='Size of cropped HR image')
    parser.add_argument('--num_threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--num_channels', type=int, default=1, help='The number of channels to super-resolve')
    parser.add_argument('--num_epochs', type=int, default=2000, help='The number of epochs to run')
    parser.add_argument('--save_epochs', type=int, default=10, help='Save trained model every this epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--Word', type=bool, default=False)
    # parser.add_argument('--ckpt_dir', type = str, default = '1024_SR_mh_checkpoint')
    # parser.add_argument("--ckpt_name", type=str, default = '1024_Net_new4')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --model_save_dir
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.img_save_dir):
        os.makedirs(args.img_save_dir)


    assert args.GT_dir, 'Error: GT_dir path is not exist.'
    assert args.LR_dir, 'Error: LR_dir path is not exist..'
    assert args.test_dataset, 'Error: test_dataset path is not exist..'
    assert args.pretrained_model, 'Error: pretrained_model path is not exist..'
    # --epoch
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

""" main """
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode=False")

    print('scale factor = ', scale_factor,\
        'crop_size = ', args.crop_size,\
        '\nlr = ', args.lr, \
        '\nGT_dir =', args.GT_dir, \
        '\nLR_dir =', args.LR_dir, \
        '\nimg_save_dir =', args.img_save_dir, \
        '\nmodel_save_dir =', args.model_save_dir,\
    )


    # model
    model = TCL_SuperResolution(args)
    model.train()




if __name__ == '__main__':

    main()

