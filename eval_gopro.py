# Input Single 

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
from gopro_dataloader import *
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

    target_rgba = mpi.mpi_lf_wrapping(rgba_sr.cuda(), depth_planes[0], v, u)
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

    with tqdm(enumerate(val_loader), total=len(val_loader), desc='Testing-{} with {}'.format(dataset, depth_network)) as vepoch:
        for i_batch, batched_inputs in vepoch:
            img = batched_inputs[0].to(device)# ** fact
            depth = batched_inputs[1].to(device)
            #print(img.shape, depth.shape)

            with torch.no_grad():
                pred_lf = np.zeros([1, 7, 7, 3, 352, 528])

                for v in range(0, 7):
                    for u in range(0, 7):
                        inputs = img

                        lf_shear_init, rgbad_init = model_fix(inputs, depth, 3-v, 3-u)
                        mask = get_mask(rgbad_init.detach(), 3-v, 3-u)
                        mask_rest = 1.0 - mask

                        lf_shear2, rgbad2 = model_vgg(inputs, depth, 3-v, 3-u)
                        final_output = mask_rest*lf_shear_init + mask*lf_shear2

                        final_output = denormalize_lf(final_output).permute([0, 3, 1, 2])
                        #print(final_output.shape)
                        pred_lf[:,v,u,:,:,:] = final_output.cpu().numpy()

                lf_paths, img_paths = utils.get_paths(save_path, i_batch, pred_lf.shape[0])

                lf_imgs = utils.lftensor2lfnp(pred_lf)
                img = denormalize_lf(img)
                inp_imgs = utils.imtensor2imnp(np.uint8(255*img.cpu()))

                for img, path in zip(inp_imgs, img_paths):
                    #print(img.shape)
                    imageio.imwrite(path, np.uint8(img))
                for lf, path in zip(lf_imgs, lf_paths):
                    #print(lf.shape, lf.max())
                    utils.save_video_from_lf(lf, path)


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    ####################################### Experiment arguments ######################################
    parser.add_argument('--results', default='results_gopro', type=str, help='directory to save results')
    
    parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loading')
    
    ######################################## Dataset parameters #######################################
    parser.add_argument('-d', '--dataset', default='GoPro', type=str, help='Dataset to train on')

    parser.add_argument('--data_path', default='/media/data/prasan/datasets/GoPro', type=str,
                        help='path to dataset')
    
    #parser.add_argument('--filenames_file', default='test_inputs/TAMULF/test_files.txt', type=str, 
    #                    help='path to the filenames text file testing')

    parser.add_argument('-dn', '--depth_network', default='DeepLens', type=str, 
                        help='depth network used for depth inputs')
    parser.add_argument('-cc', '--color_corr', default=False, action='store_true')

    args = parser.parse_args()
    args.filenames_file = f'gopro_test_inputs/{args.dataset}/test_files.txt'

    if args.results != '.' and not os.path.isdir(args.results):
        os.makedirs(args.results)

    device = torch.device('cuda:{}'.format(args.gpu))
    feature_extract = True
    batch_size = 1
    num_mpi_planes = 8
    lfsize = [352, 528, 7, 7]
    lfsize_test = [352, 528, 7, 7]
    mode = 'validation'
    #resume = True

    # create the model
    model_fix = MpiNet(ngf=32, num_mpi_planes=8, device=device)
    model_fix = model_fix.to(device=device)
    model_vgg = MpiNet(ngf=32, num_mpi_planes=8, device=device)
    model_vgg = model_vgg.to(device=device)


    exp = 'pretrain-{}'.format(args.depth_network)
    checkpoint = torch.load('checkpoints/occluded_net.pth.tar', map_location='cpu')
    model_vgg.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load('checkpoints/visible_net.pth.tar', map_location='cpu')
    model_fix.load_state_dict(checkpoint['state_dict'])

    #epoch = 10
    #exp = 'vMPI-DeepLens-14-Feb_00:07:44'
    #exp = 'vMPI-DPT-14-Feb_00:08:06'
    #checkpoint = torch.load('logs-TAMULF/{}/checkpoints/occluded_{:03d}.pt'.format(exp, epoch-1), map_location='cpu')
    #model_vgg.load_state_dict(checkpoint['state_dict'])
    #checkpoint = torch.load('checkpoints/visible_net.pth.tar', map_location='cpu')
    #model_fix.load_state_dict(checkpoint['state_dict'])

    validation_set = LightFieldDataset(args.data_path, args.filenames_file, args.depth_network, color_corr=args.color_corr, mode=mode)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False)

    save_path = os.path.join(args.results, exp, args.dataset)
    os.makedirs(save_path, exist_ok=True)

    fact = 1.0
    print(f'Factor:{fact}')
    evaluate_model(model_vgg, model_fix, args.dataset, val_loader, args.depth_network, save_path)
    