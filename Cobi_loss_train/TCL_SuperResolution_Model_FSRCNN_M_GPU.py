import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from base_networks import *
from torch.utils.data import DataLoader
# from data import get_dataset, get_test_set
from dataset_1207 import *
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
#torch.cuda.set_device(0,1,2,3,4,5,6,7)
from srlosses import CoBi_Loss, Patch_CoBi_Loss

os.environ['CUDA_VISIBLE_DEVICES']='5,6,7'

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
            self.loss_func = args.loss_func


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
        else:
            raise Exception('error: undefined Model_name')

        self.model = net(scale_factor=self.scale_factor, num_channels=1, d=56, s=12, m=4)

        
        # choose pretrained model
        if Model_index==0:
            print('no need pretrained model')
            # self.model.weight_init_fromMAT_s() #init from trained model
            self.model.weights_init_kaiming()
        elif Model_index==1:
            self.model.load_state_dict(torch.load(self.pretrained_model))
            print('\nTrained generator model is loaded:' + self.pretrained_model)
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
        

    def train(self):

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
        if self.loss_func == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_func == 'ssim':
            self.loss = pytorch_ssim.SSIM(window_size = 11)
        elif self.loss_func == 'Cobi':
            self.loss = CoBi_Loss()
            
        if self.gpu_mode:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
            self.loss.cuda()

        # load dataset
        train_data_loader = self.load_dataset(dataset='train')
        #val_data_loader = self.load_dataset(dataset='test')

        # set the logger
        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = Logger(log_dir)

        #ckpt_dir = self.ckpt_dir
        #if not os.path.exists(ckpt_dir): 
        #    os.makedirs(ckpt_dir, mode=0o777) 

        ################# Train start#################
        print('Training is started.')
        avg_loss = []
        step = 0
        self.model.train()
        ###  debug ###

        ### debug end ###
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for iter, data in enumerate(train_data_loader):
                LR = data['img_LR']
                HR = data['img_HR']

                #only use Y channel
                input_Y = LR[:, 0:1, :, :]
                #print(input_Y.shape)
                target_Y = HR[:, 0:1, :, :]
                #if self.scale_factor==4:
                #    target_Y = utils.shave(target_Y, border_size=2 * self.scale_factor) #according to size of the output image passed the network
                #elif self.scale_factor==6:
                #    target_Y = utils.shave(target_Y, border_size=2 * self.scale_factor - 1)
                #elif self.scale_factor==2:
                #    target_Y = utils.shave(target_Y, border_size=2 * self.scale_factor - 1)
                #else:
                #    target_Y = utils.shave(target_Y, border_size=2 * self.scale_factor - 2)

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
                    # target = Variable(utils.shave(target_Y, border_size=2*self.scale_factor))
                    input = Variable(input_Y)


                ############## ORIGINAL ###############
                self.optimizer.zero_grad()
                recon_image = self.model(input)
                #print(recon_image.shape)
                #print('target.shape', target.shape)
                # if self.scale_factor ==2:
                #     recon_image = recon_image[:,:,1:-1,1:-1]
                # elif self.scale_factor == 3:
                #     recon_image = recon_image[:, :, 0:-1, 0:-1]

                #### SSIM loss ##############
                # loss = 1-self.loss(recon_image, target)

                loss = self.loss(recon_image, target)

                # print loss.data
                loss.backward()
                self.optimizer.step()
                # log
                epoch_loss += loss.data
                # tensorboard logging
                logger.scalar_summary('loss', loss.data, step + 1)


                step += 1


            if epoch % self.save_epochs == 0:
                #onnx_name = 'x' + str(self.scale_factor) + '_' + self.model_name + '_epoch_' + str(epoch) + '.onnx' 
                #torch.onnx.export(self.model, input, onnx_name, export_params=True, verbose=True)

                self.save_model(epoch)

                #save_path = os.path.join(ckpt_dir, "{}_{}.pth".format(self.ckpt_name, epoch)) 
                #torch.save(self.model.state_dict(), save_path)
                
                #self.validation(epoch, val_data_loader)

            avg_loss.append(epoch_loss / len(train_data_loader))
            print("Epoch: [%2d] [%4d/%4d] loss: %.8f" % ((epoch + 1), (iter + 1), len(train_data_loader), epoch_loss))



        # Plot avg. loss
        utils.plot_loss([avg_loss], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def validation(self, epoch, val_data_loader): #input as YCbCr   to be complete
        print('Validation is started.')
        os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
        #torch.cuda.set_device(4)
        # test_data_loader = self.load_dataset(dataset='test')
        self.model.eval()
        img_num = 0
        total_loss = 0
        for _, (LR, target) in enumerate(val_data_loader):
            input_x = LR[:, 0:1, :, :]
            target_Y = target[:, 0:1, :, :]
            target_y = utils.shave(target_Y, border_size=2 * self.scale_factor)
            if self.gpu_mode:
                input = Variable(input_x.cuda())
                target = Variable(target_y.cuda())
            else:
                target = Variable(target_y)
                # target = Variable(utils.shave(target_Y, border_size=2*self.scale_factor))
                input = Variable(input_x)
                            

            #for ch in range(3):
             #   input_current = LR[:, ch:ch + 1, :, :]
              #  if ch == 0:
               #     target_0 = target[:, ch:ch + 1, :, :]
                #    target_y = utils.shave(target_0, border_size=2 * self.scale_factor)
                 #   if self.gpu_mode:
                  #      input = Variable(input_current.cuda())
                   #     target = Variable(target_y.cuda())
                   # else:
                    #    input = Variable(input_current)
                     #   target = Variable(target_y)

                    # prediction
            recon_imgs = self.model(input)   #inference 3 channel, to be complete
            loss = self.loss(recon_imgs, target)
            total_loss += loss

                    #recon_y = recon_imgs.detach()
                    
                    #temp_y = recon_y[0,0,:,:] * 255
                    #temp = temp_y.cpu().numpy()
                    #temp1 = np.clip(temp, 0, 255)
                    #out_y = Image.fromarray(np.uint8(temp1))
                
                # if ch == 1:
                # m = nn.Upsample(self.scale_factor, mode='bicubic')
                # yb_tensor = utils.shave(m(input_current), border_size=2 * self.scale_factor
        
        avg_loss = total_loss / len(val_data_loader)
        print('avg_loss = ', avg_loss)

            #     if ch == 0:
            #         recon_img_3ch = torch.from_numpy(
            #             numpy.zeros((1, 3, recon_imgs.shape[2], recon_imgs.shape[3]), dtype='f'))

            #     recon_img_3ch[:, ch, :, :] = recon_imgs.data[0, :, :, :]

            # for i, recon_img in enumerate(recon_img_3ch):  # for each image in batch size
            #     img_num += 1

            #     recon_img = recon_img_3ch[i]
            #     recon_img *= 255.0
            #     output_R = recon_img[0,:,:] + 1.402 * (recon_img[2,:,:] - 128)
            #     output_G = recon_img[0,:,:] - 0.344136 * (recon_img[1,:,:] - 128) - 0.714136 * (recon_img[2,:,:] - 128)
            #     output_B = recon_img[0,:,:] + 1.772 * (recon_img[1,:,:] - 128)

            #     output_all = numpy.zeros(( 3, output_R.shape[0], output_R.shape[1]))
            #     output_all[0, :, :] = output_R
            #     output_all[1, :, :] = output_G
            #     output_all[2, :, :] = output_B
            #     output_all = torch.from_numpy(output_all)
            #     recon_img = output_all.clamp(0, 255).numpy().transpose(1, 2, 0).astype(numpy.uint8)

       # scipy.misc.imsave(self.img_save_dir + '/img_only_y' + str(img_num) + '_' + str(self.scale_factor) + 'x_' + str(epoch)+'LR_'+str(self.lr)+'.png', out_y)


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

                if ch==0:
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
            torch.save(self.model.state_dict(), self.save_dir + '/lr' + '_'+str(self.lr) + '_x' + str(self.scale_factor) + '_' + self.model_name + '_epoch_%d.pkl' % epoch)
           # torch.save(self.model.state_dict(), self.save_dir + 'lr' + '_' +str(self.lr) + '_x' + str(self.scale_factor) + '_' + self.model_name + '_epoch_%d.pth' % epoch)
        else:
            torch.save(self.model.state_dict(), self.save_dir + '/lr' + self.model_name + '_param.pkl')
           # torch.save(self.model.state_dict(), self.save_dir + '/lr' + self.model_name + '_param.pth')

        print('Trained model is saved.')

