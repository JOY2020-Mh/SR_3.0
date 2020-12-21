import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from base_networks import *
from torch.utils.data import DataLoader
# from data import get_dataset, get_test_set
from dataset_1215 import *
import utils
from logger import Logger
from torchvision.transforms import *
import scipy.io
import numpy
import scipy.misc
import time
import pytorch_ssim
import random
from PIL import Image
import torch
import PIL.Image as pil_image
from os import listdir
from os.path import join
#torch.cuda.set_device(0,1,2,3,4,5,6,7)
from srlosses import CoBi_Loss, Patch_CoBi_Loss
from loss import *
from utils_FSRCNN import convert_ycbcr_to_rgb, preprocess, calc_psnr

#os.environ['CUDA_VISIBLE_DEVICES']='7'

import torch._utils
import torch
import torch.onnx
from torch.autograd import Variable
import numpy as np
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class TCL_SuperResolution(object):
    def __init__(self, args):
        # parameters
        self.is_train = args.is_train
        if self.is_train == True:
            self.GT_dir = args.GT_dir
            self.LR_dir = args.LR_dir
            self.save_inImg = args.save_inImg
            self.num_epochs = args.num_epochs
            self.save_epochs = args.save_epochs
            self.batch_size = args.batch_size
            self.crop_size = args.crop_size
            self.num_threads = args.num_threads
            self.lr = args.lr
            self.save_dir = args.model_save_dir
            self.img_save_dir = args.img_save_dir
            self.Model_index = args.Model_index
            #self.loss_func = args.loss_func
            self.vgg_indices = args.vgg_indices
            self.lr_decrease_mode = args.lr_decrease_mode
            self.lr_decrease_factor = args.lr_decrease_factor
            self.lr_decrease_epoch = args.lr_decrease_epoch
            self.lr_decrease_iter = args.lr_decrease_iter


        self.inYCbCr = args.inYCbCr
        self.num_channels = args.num_channels
        self.model_name = args.model_name
        self.test_dataset = args.test_dataset
        self.test_GT = args.test_GT
        self.is_LR = args.is_LR
        self.scale_factor = args.scale_factor
        self.pretrained_model = args.pretrained_model
        self.gpu_mode = args.gpu_mode
        #self.ckpt_dir = args.ckpt_dir
        #self.ckpt_name = args.ckpt_name
        self.image_test_HR_dir = args.image_test_HR_dir
        self.image_test_LR_dir = args.image_test_LR_dir



        # self.pretrained_model_name = args.pretrained_model_name

  
    def load_dataset(self, dataset):

        if dataset == 'train':
            print('Loading train datasets ...')
            # NOTE: TrainDatasetFromFolder input 2 dir for easy pairing, TestDatasetFromFolder_GT input 1 dir for easy debugging
            train_set = TrainDatasetFromFolder(self.GT_dir, self.LR_dir, crop_size=self.crop_size, scale_factor=self.scale_factor, is_gray=False, save_inImg=self.save_inImg, inYCbCr=self.inYCbCr)
            train_data_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers= self.num_threads, pin_memory=True)

            return train_data_loader

        elif dataset == 'test':
            print('Loading test datasets...', self.test_dataset)
            test_data_loader = []
            #HRLR pair(test_GT, test_dataset) or only HR(test_dataset) or only LR(test_dataset)
            if self.test_GT and self.test_dataset:
                test_set_GT = TestDatasetFromFolder_GT(self.test_GT, inYCbCr=self.inYCbCr, scale_factor=self.scale_factor, is_LR=False)
                test_set_LR = TestDatasetFromFolder_LR(self.test_dataset, inYCbCr=self.inYCbCr, scale_factor=self.scale_factor, is_LR=True)
            else:
                test_set_GT = TestDatasetFromFolder_GT(self.test_dataset, inYCbCr=self.inYCbCr, scale_factor=self.scale_factor, is_LR=self.is_LR)
                test_set_LR = TestDatasetFromFolder_LR(self.test_dataset, inYCbCr=self.inYCbCr, scale_factor=self.scale_factor, is_LR=self.is_LR)

            for ind_image in range(len(test_set_GT.image_filenames)):
                test_target = test_set_GT.__getitem__(ind_image)
                test_input = test_set_LR.__getitem__(ind_image)
                test_input = test_input.unsqueeze(0)
                test_target = test_target.unsqueeze(0)
                test_data_loader.append((test_input, test_target))
            return test_data_loader


    def Choose_Model(self, Model_index):
        # choose model structure
        if self.model_name == 'Net':
            from network import Net as net
        elif self.model_name == 'Net_new3':
            from network import Net_new3 as net
        elif self.model_name == 'Net_new4':
            from network import Net_new4 as net
        elif self.model_name == 'FSRCNN':
            from network import FSRCNN as net
        elif self.model_name == 'FSRCNN_d1':
            from network import FSRCNN_d1 as net
        else:
            raise Exception('error: undefined Model_name')

        self.model = net(scale_factor=self.scale_factor, num_channels=1, d=56, s=12, m=4)

        
        # choose pretrained model
        if Model_index==0:
            print('no need pretrained model')
            # self.model.weight_init_fromMAT_s() #init from trained model
            #self.model.weights_init_kaiming()
        elif Model_index==1:
            state_dict = torch.load(self.pretrained_model)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
            #print(k)
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            self.model.load_state_dict(new_state_dict)
            #self.model.load_state_dict(torch.load(self.pretrained_model))
            print('Trained generator model is loaded:', self.pretrained_model)
        elif Model_index==2:
            self.model.weight_init()
            pretrained_dict = torch.load(self.pretrained_model)
            model_dict = self.model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print('model partially initialized')
        else:
            raise Exception('error: undefined Model_index')



     #Learning rate decrease
    
    def adjust_learning_rate(self, epoch, iteration):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if self.lr_decrease_mode == 'epoch':
            lr = self.lr * (self.lr_decrease_factor ** (epoch // self.lr_decrease_epoch))
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
        if self.lr_decrease_mode == 'iter':
            lr = self.lr * (self.lr_decrease_factor ** (iteration // self.lr_decrease_iter))
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
         
    def train(self,args):

        # load networks************************************************************************
        self.Choose_Model(self.Model_index)
        utils.print_network(self.model)

        # optimizer
        self.optimizer = optim.Adam([
            {'params': self.model.first_part.parameters()},
            {'params': self.model.mid_part.parameters()},
            {'params': self.model.last_part.parameters(), 'lr': self.lr * 0.1}
        ], lr=self.lr)

        

        # self.momentum = 0.9
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.model.float()
        # loss function

        if 'l1' in args.loss:
            criterion_L1 = torch.nn.L1Loss().cuda()
    
        if 'ssim' in args.loss:
            ssimLoss = SSIM().cuda()    

        if 'tv' in args.loss:
            totalvar_loss = tv_loss().cuda()

        if 'color' in args.loss:
            csColorLoss = color_loss().cuda()

        if 'grad' in args.loss:
            gradLoss = GradientLoss().cuda()

        if 'vgg' in args.loss:
            vggLoss = VGGLoss(args).cuda().eval()

        if 'mse' in args.loss:
            mseLoss = nn.MSELoss().cuda()

        if 'Cobi' in args.loss:
            CoBiLoss = CoBi_Loss().cuda()


        if args.multi_gpu:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
        else:
            self.model.cuda()

        # load dataset
        train_data_loader = self.load_dataset(dataset='train')
        #val_data_loader = self.load_dataset(dataset='test')

        # set the logger
        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = Logger(log_dir)

        ################# Train start#################
        print('Training is started.')
        avg_loss = []
        step = 0
        self.model.train()
        ###  debug ###

        ### debug end ###
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            avg_l1_loss = 0
            avg_tv_loss = 0
            avg_cs_ColorLoss=0
            avg_grad_loss = 0
            avg_vgg_loss=0
            avg_mse_loss = 0
            avg_cobi_loss = 0
            avg_ssim_loss = 0

            for iter, data in enumerate(train_data_loader):
                LR = data['img_LR']
                HR = data['img_HR']

                #only use Y channel
                input_Y = LR[:, 0:1, :, :]
                #print(input_Y.shape)
                target_Y = HR[:, 0:1, :, :]

                if self.save_inImg == True:  #save the net input image
                    saveinY = (input_Y.numpy()[0, :, :, :].transpose(1, 2, 0)*255).astype(numpy.uint8)
                    scipy.misc.imsave('lrin.png', saveinY[:, :, 0]);
                    savetarY = (target_Y.numpy()[0, :, :, :].transpose(1, 2, 0)*255).astype(numpy.uint8)
                    scipy.misc.imsave('tarin.png', savetarY[:, :, 0]);


                if self.gpu_mode:
                    target = Variable(target_Y.cuda())
                    input = Variable(input_Y.cuda())
                else:
                    target = Variable(target_Y)
                    input = Variable(input_Y)


                ############## ORIGINAL ###############
                
                recon_image = self.model(input)
                # overall loss
                loss = 0

                disp_Pixellevel_L1_Loss = 0
                disp_tot_v_loss = 0
                disp_cs_ColorLoss = 0
                disp_grad_loss = 0
                disp_vgg_loss = 0
                disp_ssim_loss = 0
                disp_mse_loss = 0
                disp_Cobi_loss = 0
               
                # L1 Loss
                if 'l1' in args.loss:
                    Pixellevel_L1_Loss = args.lambda_l1 * criterion_L1(recon_image, target)
                    loss += Pixellevel_L1_Loss
                    avg_l1_loss += Pixellevel_L1_Loss.item()
                    disp_Pixellevel_L1_Loss = Pixellevel_L1_Loss.item()
                # tv Loss
                if 'tv' in args.loss:
                    tot_v_loss = args.lambda_tv * totalvar_loss(recon_image)
                    loss += tot_v_loss
                    avg_tv_loss += tot_v_loss.item()
                    disp_tot_v_loss = tot_v_loss.item()


                # color loss
                if 'color' in args.loss:
                    cs_ColorLoss = args.lambda_color * csColorLoss(recon_image, target)
                    loss += cs_ColorLoss
                    avg_cs_ColorLoss += cs_ColorLoss.item()
                    disp_cs_ColorLoss = cs_ColorLoss.item()

                # gradient loss
                if 'grad' in args.loss:
                    grad_loss = args.lambda_grad * gradLoss(recon_image, target)
                    loss += grad_loss
                    avg_grad_loss += grad_loss.item()
                    disp_grad_loss = grad_loss.item()

                # vgg loss
                if 'vgg' in args.loss:

                    if args.num_channels == 1:
                        vgg_loss = vggLoss(recon_image.clamp(0, 1).expand(-1,3,-1,-1), target.clamp(0, 1).expand(-1,3,-1,-1))
                    else:
                        vgg_loss = vggLoss(recon_image.clamp(0, 1), target.clamp(0, 1))
                    loss += vgg_loss
                    avg_vgg_loss += vgg_loss.item()
                    disp_vgg_loss = vgg_loss.item()
           
                # ssim loss
                if 'ssim' in args.loss:
                    ssim_loss = args.lambda_ssim * (1 - ssimLoss(recon_image, target))
                    loss += ssim_loss
                    avg_ssim_loss += ssim_loss.item()
                    disp_ssim_loss = ssim_loss.item()

                # ssim loss
                if 'mse' in args.loss:
                    mse_loss = args.lambda_mse * mseLoss(recon_image, target)
                    loss += mse_loss
                    avg_mse_loss += mse_loss.item()
                    disp_mse_loss = mse_loss.item()
                
                # Cobi loss
                if 'cobi' in args.loss:
                    cobi_loss = args.lambda_cobi * CoBiLoss(recon_image, target)
                    loss += cobi_loss
                    avg_cobi_loss += cobi_loss.item()
                    disp_cobi_loss = cobi_loss.item()

                

                # print loss.data
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print('lr', self.lr)

                # Determine approximate time left
                iters_done = epoch * len(train_data_loader) + iter
                #iters_left = args.epochs * len(dataloader) - iters_done
                #time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
                #prev_time = time.time()
                #print('loss:', loss.data)
                # log
                epoch_loss += loss.data
                # tensorboard logging
                logger.scalar_summary('loss', loss.data, step + 1)
                step += 1
                ###test validation
                #self.validation()


            print("\r[Epoch %d/%d] [avg_Pixellevel_L1 Loss: %.4f] [avg_color_loss: %.4f] [avg_ssimLoss: %.4f] [avg_grad_Loss: %.4f] [avg_VGG_Loss: %.4f] [avg_TV_Loss: %.4f] [avg_mse Loss: %.4f]" %
                    ((epoch + 1), self.num_epochs, avg_l1_loss/len(train_data_loader), avg_cs_ColorLoss/len(train_data_loader), avg_ssim_loss/len(train_data_loader), avg_grad_loss/len(train_data_loader), avg_vgg_loss/len(train_data_loader), avg_tv_loss/len(train_data_loader), avg_mse_loss/len(train_data_loader)))
            self.adjust_learning_rate((epoch + 1), (iters_done + 1))

            self.save_model(epoch)
            self.validation()
        
            avg_loss.append(epoch_loss / len(train_data_loader))
            print("Epoch: [%2d] [%4d/%4d] loss: %.8f" % ((epoch + 1), (iter + 1), len(train_data_loader), epoch_loss))

        # Plot avg. loss
        utils.plot_loss([avg_loss], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def validation(self): #input as YCbCr   to be complete

        def is_image_file(filename):
            return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jepg', '.bmp'])

        print('Validation is started.')
        # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
        #torch.cuda.set_device(4)
        # test_data_loader = self.load_dataset(dataset='test')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        
        ###Set SSIM and PSNR 
        ssim = pytorch_ssim.SSIM(window_size = 11) ###not ssim loss but ssim value
        # ssim = pytorch_ssim.SSIM(window_size = 11) ###not ssim loss but ssim value
        total_ssim = 0
        total_psnr = 0

        image_HR_dir = self.image_test_HR_dir
        image_LR_dir = self.image_test_LR_dir

        image_HR_filenames = [join(image_HR_dir, x) for x in sorted(listdir(image_HR_dir)) if is_image_file(x)]
        image_LR_filenames = [join(image_LR_dir, x) for x in sorted(listdir(image_LR_dir)) if is_image_file(x)]
        file_num = len(image_HR_filenames)
        for idx in range(file_num):
            # print(idx)
            image = pil_image.open(image_LR_filenames[idx]).convert('RGB')
            hr = pil_image.open(image_HR_filenames[idx]).convert('RGB')
            lr = image
            bicubic = lr.resize((lr.width * self.scale_factor, lr.height * self.scale_factor), resample=pil_image.BICUBIC)
            lr, _ = preprocess(lr, device)
            hr, _ = preprocess(hr, device)
            _, ycbcr = preprocess(bicubic, device)

            with torch.no_grad():
                preds = self.model(lr).clamp(0.0, 1.0)
                
                ### To calulate psnr and ssim 
                psnr = calc_psnr(hr, preds)
                # print('psnr', psnr)
                ssim_value = 1 - ssim(hr, preds)
                # print('ssim_value', ssim_value)
                total_ssim += ssim_value
                total_psnr += psnr

        print('eval psnr: {:.2f}'.format(total_psnr/file_num),'eval ssim: {:.2f}'.format(total_ssim/file_num))
       

    def test(self):

        self.Choose_Model(Model_index=1)


        print('Test is started.')
        # load dataset
        Duration_ALL = 0
        NUM = 0
        test_data_loader = self.load_dataset(dataset='test')
        self.model.eval()

        for ind_image, (LR, target) in enumerate(test_data_loader):
            ### ONE Channel ###
            start_time = time.time()
            for ch in range(3):
                # input_current = torch.from_numpy(numpy.zeros((1,1,input.shape[2],input.shape[3]),dtype=float))
                input_current = LR[:,ch:ch+1,:,:]
                # print "input_current"
                # print input_current.shape
                if self.gpu_mode:
                    input = Variable(input_current.cuda())
                else:
                    input = Variable(input_current)
                # prediction
                recon_imgs = self.model(input)
                recon_img_3ch = torch.from_numpy(numpy.zeros((1,3,recon_imgs.shape[2],recon_imgs.shape[3]),dtype='f'))
                recon_img_3ch[:,ch,:,:]=recon_imgs.data[0,:,:,:]


            Duration = time.time() - start_time
            Duration_ALL = Duration + Duration_ALL
            NUM = NUM + 1

            save_path = './Results_x'+str(self.scale_factor)+'/'+self.model_name+'/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # bc_img = utils.shave(utils.img_interp(input[0], self.scale_factor),border_size=2 * self.scale_factor)
            # bc_img *= 255.0
            # bc_img = bc_img.clamp(0, 255).numpy().transpose(1, 2, 0).astype(numpy.uint8)
            # scipy.misc.imsave(save_path+str(ind_image)+'_bc_x'+str(self.scale_factor)+'.png', bc_img)
            ## groundTruth
            # gt_img = utils.shave(target[0], border_size=2 * self.scale_factor)
            #
            # gt_img *= 255.0
            # gt_img = gt_img.clamp(0, 255).numpy().transpose(1, 2, 0).astype(numpy.uint8)
            #
            # # scipy.misc.imsave(save_path+str(ind_image)+'_GT_x'+str(self.scale_factor)+'.png', gt_img)
            #
            # ## lr_img
            # input_3d = input[0]
            # lr_img = utils.shave(input_3d, border_size=2)
            # lr_img *= 255.0
            # lr_img = lr_img.clamp(0, 255).numpy().transpose(1, 2, 0).astype(numpy.uint8)
            # scipy.misc.imsave(save_path+str(ind_image)+'_LR_x'+str(self.scale_factor)+'.png', lr_img)


            ## recon_img
            recon_img_3ch *= 255.0
            recon_img = recon_img_3ch[0].clamp(0, 255).numpy().transpose(1, 2, 0).astype(numpy.uint8)

            print('./Results_x'+str(self.scale_factor)+'/'+str(ind_image)+'_x'+str(self.scale_factor)+'.png')
            scipy.misc.imsave(save_path+str(ind_image)+'_SR_x'+str(self.scale_factor)+'_s.png', recon_img)
            # recon_img=target.numpy()
            # psnr = nn.MSELoss(recon_img, target)
            # print("psnr: %d ," % psnr)

        print('average time is :', Duration_ALL/NUM)



    def save_model(self, epoch=None):
        if epoch is not None:
            torch.save(self.model.state_dict(), self.save_dir + '/' + self.save_dir + str(self.scale_factor) + 'x' + '_' + self.model_name + '_epoch_%d.pkl' % epoch)
           # torch.save(self.model.state_dict(), self.save_dir + 'lr' + '_' +str(self.lr) + '_x' + str(self.scale_factor) + '_' + self.model_name + '_epoch_%d.pth' % epoch)
        else:
            torch.save(self.model.state_dict(), self.save_dir + '/lr' + self.model_name + '_param.pkl')
           # torch.save(self.model.state_dict(), self.save_dir + '/lr' + self.model_name + '_param.pth')

        print('Trained model is saved.')
