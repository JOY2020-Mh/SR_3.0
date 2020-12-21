
is_train = True
scale_factor = 4
#Model_index = 2   #0 for no loading pretrained model, 1 for loading pretrained model
#model_name = '4x/visidon_style/vistyle_day_v2.pkl'
model_name = 'Net'
print ('scale_factor = %d' %scale_factor)
from TCL_SuperResolution_Model_our_FSRCNN_1220_valid import TCL_SuperResolution

import torch
import os, argparse


"""parsing and configuration"""

def parse_args():
    desc = "PyTorch implementation of S R collections"
    parser = argparse.ArgumentParser(description=desc)
    ####about GPU 
    parser.add_argument('--multi_gpu', type = bool, default = True, help = 'True for more than 1 GPU')
    parser.add_argument('--single_gpu_ids', type = str, default = '0', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    ####about learning rate
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 100, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 200000, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    #############################################################################################################
    
    parser.add_argument('--is_train', type=bool, default=is_train)
    parser.add_argument('--model_name', type=str, default='FSRCNN',
                        choices=['Net', 'Net_new4', 'Net_new3','FSRCNN','FSRCNN_d1','FSRCNN_d2'], help='The type of model')
    parser.add_argument('--scale_factor', type=int, default=scale_factor, help='Size of scale factor')
    parser.add_argument('--Model_index', type=int, default=0)  ### 0 is model used until V1.6. Rest represent the update index.
    #parser.add_argument('--loss_func', type=str, default='vgg') ##mse, ssim, vgg
    #parser.add_argument('--vgg_indices', nargs="+", type=int, default=[21], help = 'vgg loss layer e.g. 2 7 12 21 30')
    #parser.add_argument('--lambda_vgg', nargs="+", type=float, default=[0.5], help = 'weighting of total variation loss e.g. 0.05 0.05 0.05 0.1 0.5')
    parser.add_argument('--GT_dir', type=str, default='./SR_train/HR')
    parser.add_argument('--LR_dir', type=str, default='./SR_train/LR')
    parser.add_argument('--image_test_HR_dir', type = str, default='./0_Image_test_and_validation/RealSR_test_images/HR')
    parser.add_argument('--image_test_LR_dir', type = str, default = './0_Image_test_and_validation/RealSR_test_images/LR')
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
    parser.add_argument('--lr', type=float, default=3 * 0.0001)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--Word', type=bool, default=False)

    # loss parameters
    parser.add_argument('--loss', nargs="+", type=str, default=['vgg','mse'], help = 'loss function: l1, ssim, gradient loss, vgg loss, color loss, tv loss mse loss cobi')
    parser.add_argument('--lambda_l1', type=float, default=1, help = 'weighting of l1 loss')
    parser.add_argument('--lambda_ssim', type=float, default=1, help = 'weighting of ssim loss')
    parser.add_argument('--lambda_grad', type=float, default=0.1, help = 'weighting of gradient loss')
    parser.add_argument('--lambda_color', type=float, default=0.1, help = 'weighting of color loss')
    parser.add_argument('--lambda_tv', type=float, default=0.1, help = 'weighting of total variation loss')
    parser.add_argument('--lambda_mse', type=float, default=0.5, help = 'weighting of mse loss')
    parser.add_argument('--lambda_cobi', type=float, default=0.5, help = 'weighting of mse loss')
    parser.add_argument('--vgg_indices', nargs="+", type=int, default=[2,7,21], help = 'vgg loss layer e.g. 2 7 12 21 30')
    parser.add_argument('--lambda_vgg', nargs="+", type=float, default=[0.05,0.1,0.5], help = 'weighting of total variation loss e.g. 0.05 0.05 0.05 0.1 0.5')
    


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

    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if args.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (args.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.single_gpu_ids
        print('Single-GPU mode')

    # model
    model = TCL_SuperResolution(args)
    model.train(args)




if __name__ == '__main__':

    main()

