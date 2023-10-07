#!/usr/bin/env python

# Single Input

import os
import argparse
import random
import time
import numpy as np
import math
from tqdm import tqdm
from PIL import Image
import skimage
import skimage.transform
import matplotlib.pyplot as plt
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as ttf
import torch.optim as optim
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

import mpi
from model import *
from dataloader import *
from model_io import *
from vgg import vgg_loss, weighted_vgg_loss
from loss import SSIMLoss
import tensor_ops as utils
from utils import RunningAverage


def lock_mpi_net(model):
    for name, param in model.named_parameters():
        if param.requires_grad and 'inp' not in name:
            param.requires_grad = False


def gradient_hook_deco(mask):
    def hook(grad):
        return mask*grad
    return hook


def get_mask(rgbad, v, u):
    color_layers = rgbad[:, :, :, :, :3]
    alpha_layers = rgbad[:, :, :, :, 3:4]
    depth_layers = rgbad[:, :, :, :, 4]
    depth_planes = depth_layers.mean(1).mean(1)

    alpha = alpha_layers[:,:,:,:,0]
    a_ini = alpha.permute([0, 3, 1, 2])

    rgba_sr = torch.tensor(rgbad[...,:4])
    for i in np.arange(0, num_mpi_planes):
        # calculate a_occ_i:
        for j in range(i+1, num_mpi_planes):
            if j == i+1:
                a_occ_i = a_ini[:,j:j+1,:,:].clone().detach().requires_grad_(True)
            else:
                a_occ_i = a_occ_i*(1-a_ini[:,j:j+1,:,:]) + a_ini[:,j:j+1,:,:]
        if i+1 == num_mpi_planes:
            a_occ_i = torch.zeros_like(a_ini[:,0:1,:,:], requires_grad=True)

        a_occ_i = a_occ_i.permute([0, 2, 3, 1])
        rgba_sr[:,:,:,i,:] = a_ini[:,i:i+1,:,:].permute([0,2,3,1])*(1-a_occ_i)

    target_rgba = mpi.mpi_lf_wrapping(rgba_sr.to(device), depth_planes[0], v, u, device=device)
    target_alpha = target_rgba[:,:,:,:,3:]
    target_alpha_sum = torch.sum(target_alpha, dim=0)
    target_alpha_sum = torch.clamp(target_alpha_sum, 0, 1)
    weight = 1. - target_alpha_sum
    weight[weight<0.2] = 0

    return weight.to(device)
    

def change_param_to_eval():
    global lfsize, batch_size
    lfsize[0] = lfsize_test[0]
    lfsize[1] = lfsize_test[1]
    batch_size = 1


def calculate_psnr(img1, img2):
    with torch.no_grad():
        try:
            img1 = 255 * img1.cpu().numpy()
        except:
            img1 = 255 * img1
        try:
            img2 = 255 * img2.cpu().numpy()
        except:
            img2 = 255 * img2
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    with torch.no_grad():
        SSIM = SSIMLoss()
        N, V, _, C, H, W = img1.shape
        img1 = torch.tensor(img1.reshape(N*V*V, C, H, W))
        img2 = torch.tensor(img2.reshape(N*V*V, C, H, W))
        ssim = SSIM(img1, img2).numpy()
        return ssim


def evaluate_model(model_vgg, model_fix, dataset, val_loader, depth_network, save_path):
    model_fix.eval()
    model_vgg.eval()
    change_param_to_eval()
    psnr_avg = RunningAverage()
    ssim_avg = RunningAverage()

    all_times = []

    f = open(os.path.join(save_path, 'metrics.txt'), 'w')

    with tqdm(enumerate(val_loader), total=len(val_loader), desc='Testing-{} with {}'.format(dataset, depth_network)) as vepoch:
        for i_batch, batched_inputs in vepoch:
            lf = batched_inputs[1]# ** fact
            depth = batched_inputs[2].to(device)
            img = lf[:,3,3,...]

            with torch.no_grad():
                pred_lf = np.zeros(lf.shape)
                starttime = time.time()
                for v in range(0, 7):
                    for u in range(0, 7):
                        inputs, target = lf[:,3,3,...], lf[:,v,u,...]
                        inputs, target = data_augmentation(inputs.numpy(), target.numpy(), device, mode=mode)

                        lf_shear_init, rgbad_init = model_fix(inputs, depth, 3-v, 3-u)
                        mask = get_mask(rgbad_init.detach(), 3-v, 3-u)
                        mask_rest = 1.0 - mask

                        lf_shear2, rgbad2 = model_vgg(inputs, depth, 3-v, 3-u)
                        final_output = mask_rest*lf_shear_init + mask*lf_shear2

                        final_output = denormalize_lf(final_output).permute([0, 3, 1, 2])
                        pred_lf[:,v,u,:,:,:] = final_output.cpu().numpy()
                endtime = time.time() - starttime

                # print(pred_lf.shape)

                psnr = calculate_psnr(pred_lf, lf)
                ssim = calculate_ssim(pred_lf, lf)

                psnr_avg.append(psnr)
                ssim_avg.append(ssim)

                # NOTE: Use below for getting qualitative results
                # lf_paths, img_paths = utils.get_paths(save_path, i_batch, pred_lf.shape[0])

                # lf_imgs = utils.lftensor2lfnp(pred_lf)
                # inp_imgs = utils.imtensor2imnp(np.uint8(255*img.cpu()))

                # for img, path in zip(inp_imgs, img_paths):
                #     img = np.transpose(img, (1, 2, 0))
                #     imageio.imwrite(path, np.uint8(img))
                # for lf, path in zip(lf_imgs, lf_paths):
                #     utils.save_video_from_lf(lf, path)
                
                vepoch.set_postfix(PSNR=f"{psnr_avg.get_value():0.4f}({psnr:0.4f})",
                                   SSIM=f"{ssim_avg.get_value():0.4f}({ssim:0.4f})")

                string = 'Sample {0:2d} => PSNR: {1:.4f}, SSIM: {2:.4f}, Time: {3:.3f}\n'.format(i_batch, psnr, ssim, endtime)
                f.write(string)

    avg_psnr = psnr_avg.get_value()
    avg_ssim = ssim_avg.get_value()
    avg_time = np.mean(all_times)
    string = 'Average PSNR: {0:.4f}\nAverage SSIM: {1:.4f} Average Time: {2:.3f}\n'.format(avg_psnr, avg_ssim, avg_time)
    f.write(string)
    f.close()


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Testing script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    ####################################### Experiment arguments ######################################
    parser.add_argument('--results', default='test_results_aryan', type=str, help='directory to save results')
    
    parser.add_argument('--gpu', default=2, type=int, help='which gpu to use')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loading')
    
    ######################################## Dataset parameters #######################################
    parser.add_argument('-d', '--dataset', default='TAMULF', type=str, help='Dataset to test on')

    parser.add_argument('--data_path', default='/media/data/prasan/datasets/LF_datasets', type=str,
                        help='path to dataset')
    
    #parser.add_argument('--filenames_file', default='test_inputs/TAMULF/test_files.txt', type=str, 
    #                    help='path to the filenames text file testing')

    # options: DeepLens | DPT | Unimatch
    parser.add_argument('-dn', '--depth_network', default='DPT', type=str, 
                        help='depth network used for depth inputs')
    parser.add_argument('-cc', '--color_corr', default=False, action='store_true')
    parser.add_argument('--get_model_size', default=False, action='store_true')

    args = parser.parse_args()

    args.filenames_file = f'aryan_test_inputs/{args.dataset}/test_files.txt'

    if args.results != '.' and not os.path.isdir(args.results):
        os.makedirs(args.results)

    device = torch.device('cuda:{}'.format(args.gpu))
    feature_extract = True
    batch_size = 1
    num_mpi_planes = 8

    # our: 256x192 | Govindrajan: 176x264 | Li: 192x192
    # Random: 384x528 (to introduce another baseline) | 480p-SD: 480x640
    resH = [256, 176, 192, 352, 480]
    resW = [192, 264, 192, 528, 640]

    for i in range(len(resH)):
        lfsize = [resH[i], resW[i], 7, 7]
        lfsize_test = [resH[i], resW[i], 7, 7]
        mode = 'validation'
        #resume = True

        # create the model
        model_fix = MpiNet(ngf=32, num_mpi_planes=8, device=device)
        model_fix = model_fix.to(device=device)
        model_vgg = MpiNet(ngf=32, num_mpi_planes=8, device=device)
        model_vgg = model_vgg.to(device=device)


        exp = f'{resH[i]}x{resW[i]}_{args.depth_network}'

        if args.depth_network == 'DPT':
            checkpoint = torch.load('logs-TAMULF+Stanford(all_train)/vMPI_trainBoth-29-Sep_12:28:16/checkpoints/occluded_001.pt', map_location='cpu')
            checkpoint_vgg = torch.load('logs-TAMULF+Stanford(all_train)/vMPI_trainBoth-29-Sep_12:28:16/checkpoints/visible_001.pt', map_location='cpu')
        else:
            checkpoint = torch.load('checkpoints/occluded_net.pth.tar', map_location='cpu')
            checkpoint_vgg = torch.load('checkpoints/visible_net.pth.tar', map_location='cpu')

        model_fix.load_state_dict(checkpoint['state_dict'])
        model_vgg.load_state_dict(checkpoint_vgg['state_dict'])

        if args.get_model_size:
            param_size = 0
            for param in model_fix.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model_fix.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_all_mb = (param_size + buffer_size) / 1024**2

            print('model-fix size: {:.3f}MB'.format(size_all_mb))
            total_params = int(sum(p.numel() for p in model_fix.parameters()))
            print('total_params: {:.3f}'.format(total_params))

            #######################

            param_size = 0
            for param in model_vgg.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model_vgg.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_all_mb = (param_size + buffer_size) / 1024**2

            print('model-vgg size: {:.3f}MB'.format(size_all_mb))
            total_params = int(sum(p.numel() for p in model_vgg.parameters()))
            print('total_params: {:.3f}'.format(total_params))
            exit()


        validation_set = LightFieldDataset(args.data_path, args.filenames_file, 
                                           args.depth_network, color_corr=args.color_corr, 
                                           mode=mode, lfsize_test=lfsize_test)
        val_loader = DataLoader(validation_set, batch_size=1, shuffle=False)

        save_path = os.path.join(args.results, exp, args.dataset)
        os.makedirs(save_path, exist_ok=True)

        fact = 1.0
        # print(f'Factor:{fact}')
        evaluate_model(model_vgg, model_fix, args.dataset, val_loader, args.depth_network, save_path)
    