#!/usr/bin/env python
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
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as ttf
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

import wandb


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


def change_param_to_train():
    global lfsize, batch_size, mode
    lfsize[0] = lfsize_train[0]
    lfsize[1] = lfsize_train[1]
    mode = 'train'


def train_fix_model(model_fix, train_loader, optimizer, epoch, max_epoch):
    model_fix.train()
    change_param_to_train()
    loss_avg = RunningAverage()
    psnr_avg = RunningAverage()

    iters = epoch * len(train_loader)
    with tqdm(enumerate(train_loader), total=len(train_loader), desc='Epoch: {}/{}. Loop: Train Fix'.format(epoch+1, max_epoch)) as tepoch:
        for i_batch, batched_inputs in tepoch:
        #for i_batch, batched_inputs in enumerate(train_loader):
            pos = [(0, 0), (0, 6), (6, 0), (6, 6), (3, 3)]
            c = 4#random.randint(0, 4)
            v = random.randint(0, 6)
            u = random.randint(0, 6)

            labels = batched_inputs[1]
            inputs = labels[:,pos[c][0],pos[c][1],:,:,:]
            target = labels[:,v,u,:,:,:]

            # data augmentation
            inputs, target = data_augmentation(inputs.numpy(), target.numpy(), device, mode=mode)
            depth = batched_inputs[2].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # calculate the gradient mask
            lf_shear, rgbad = model_fix(inputs, depth, pos[c][0]-v, pos[c][1]-u)
            final_output = lf_shear

            final_output = denormalize_lf(final_output).permute([0, 3, 1, 2])
            target = denormalize_lf(target).permute([0, 3, 1, 2])
            v_loss = 1.0 * vgg_loss(normalize_batch(final_output), normalize_batch(target), device=device) + \
                     0.0 * torch.mean(torch.abs(final_output - target))
            v_loss.backward()
            optimizer.step()
                
            psnr = calculate_psnr(final_output, target)

            # statistics
            cur_loss = v_loss.item()
            loss_avg.append(cur_loss)
            psnr_avg.append(psnr)
            tepoch.set_postfix(Loss=f"{loss_avg.get_value():0.4f}({cur_loss:0.4f})",
                               PSNR=f"{psnr_avg.get_value():0.4f}({psnr:0.4f})")

            if iters%log_step == 0:
                summary_writer.add_scalar('Train Fix/Loss', cur_loss, iters)
                summary_writer.add_scalar('Train Fix/PSNR', psnr, iters)
                inputs = inputs.permute([0, 3, 1, 2])
                inputs = torchvision.utils.make_grid(denormalize_lf(inputs), nrow=4)
                summary_writer.add_image('Train Fix/Input', inputs, iters)
                depth = torchvision.utils.make_grid(depth, nrow=4)
                summary_writer.add_image('Train Fix/Depth', depth, iters)
                gt = torchvision.utils.make_grid(target, nrow=4)
                summary_writer.add_image('Train Fix/Target', gt, iters)
                out = torchvision.utils.make_grid(final_output, nrow=4)
                summary_writer.add_image('Train Fix/Output', out, iters)
            iters += 1

    return model_fix, optimizer


def train_vgg_model(model_vgg, model_fix, train_loader, optimizer2, epoch, max_epoch):
    model_fix.eval()
    model_vgg.train()
    change_param_to_train()
    loss_avg = RunningAverage()
    psnr_avg = RunningAverage()

    iters = epoch * len(train_loader)
    with tqdm(enumerate(train_loader), total=len(train_loader), desc='Epoch: {}/{}. Loop: Train Vgg'.format(epoch+1, max_epoch)) as tepoch:
        for i_batch, batched_inputs in tepoch:
            pos = [(0, 0), (0, 6), (6, 0), (6, 6), (3, 3)]
            c = 4#random.randint(0, 4)
            v = random.randint(0, 6)
            u = random.randint(0, 6)

            labels = batched_inputs[1]
            inputs = labels[:,pos[c][0],pos[c][1],:,:,:]
            target = labels[:,v,u,:,:,:]
            depth = batched_inputs[2].to(device)

            # data augmentation
            inputs, target = data_augmentation(inputs.numpy(), target.numpy(), device, mode=mode)
            
            # zero the parameter gradients
            optimizer2.zero_grad()
            
            # calculate the gradient mask
            lf_shear_init, rgbad_init = model_fix(inputs, depth, pos[c][0]-v, pos[c][1]-u)
            mask = get_mask(rgbad_init.detach(), pos[c][0]-v, pos[c][1]-u)
            mask_rest = 1.0 - mask

            lf_shear2, rgbad2 = model_vgg(inputs, depth, pos[c][0]-v, pos[c][1]-u)
            final_output = mask_rest*lf_shear_init + mask*lf_shear2

            hook = lf_shear2.register_hook(gradient_hook_deco(mask))
            final_output = denormalize_lf(final_output).permute([0, 3, 1, 2])
            target = denormalize_lf(target).permute([0, 3, 1, 2])
            v_loss = 1.0 * vgg_loss(normalize_batch(final_output), normalize_batch(target), device=device) + \
                     0.0 * torch.mean(torch.abs(final_output - target))
            v_loss.backward()
            hook.remove()
            optimizer2.step()
            
            psnr = calculate_psnr(final_output, target)

            # statistics
            cur_loss = v_loss.item()
            loss_avg.append(cur_loss)
            psnr_avg.append(psnr)
            tepoch.set_postfix(Loss=f"{loss_avg.get_value():0.4f}({cur_loss:0.4f})",
                               PSNR=f"{psnr_avg.get_value():0.4f}({psnr:0.4f})")

            if iters%log_step == 0:
                summary_writer.add_scalar('Train VGG/Loss', cur_loss, iters)
                summary_writer.add_scalar('Train VGG/PSNR', psnr, iters)
                inputs = inputs.permute([0, 3, 1, 2])
                inputs = torchvision.utils.make_grid(denormalize_lf(inputs), nrow=4)
                summary_writer.add_image('Train VGG/Input', inputs, iters)
                depth = torchvision.utils.make_grid(depth, nrow=4)
                summary_writer.add_image('Train VGG/Depth', depth, iters)
                gt = torchvision.utils.make_grid(target, nrow=4)
                summary_writer.add_image('Train VGG/Target', gt, iters)
                out = torchvision.utils.make_grid(final_output, nrow=4)
                summary_writer.add_image('Train VGG/Output', out, iters)
                mask = torchvision.utils.make_grid(mask.permute([0, 3, 1, 2]), nrow=4)
                summary_writer.add_image('Train VGG/Mask', mask, iters)
            iters += 1

    return model_vgg, optimizer2


def change_param_to_eval():
    global lfsize, batch_size, mode
    lfsize[0] = lfsize_test[0]
    lfsize[1] = lfsize_test[1]
    mode = 'validation'


def evaluate_model(model_vgg, model_fix, val_loader, epoch, max_epoch):
    model_fix.eval()
    model_vgg.eval()
    change_param_to_eval()
    psnr_avg = RunningAverage()
    ssim_avg = RunningAverage()

    iters = epoch * len(val_loader)
    with tqdm(enumerate(val_loader), total=len(val_loader), desc='Epoch: {}/{}. Loop: Validation'.format(epoch+1, max_epoch)) as vepoch:
        for i_batch, batched_inputs in vepoch:
            lf = batched_inputs[1]# ** fact
            depth = batched_inputs[2].to(device)
            img = lf[:,3,3,...]

            with torch.no_grad():
                pred_lf = np.zeros(lf.shape)

                for v in range(0, 7):
                    for u in range(0, 7):
                        inputs, target = lf[:,3,3,...], lf[:,v,u,...]
                        inputs, target = data_augmentation(inputs.numpy(), target.numpy(), device, mode=mode)

                        lf_shear_init, rgbad_init = model_fix(inputs, depth, 3-v, 3-u)
                        mask = get_mask(rgbad_init.detach(), 3-v, 3-u)
                        mask_rest = 1.0 - mask

                        lf_shear2, rgbad2 = model_vgg(inputs, depth, 3-v, 3-u)
                        final_output = mask_rest*lf_shear_init + mask*lf_shear2
                        #final_output = lf_shear_init

                        final_output = denormalize_lf(final_output).permute([0, 3, 1, 2])
                        pred_lf[:,v,u,:,:,:] = final_output.cpu().numpy()
                        
                psnr = calculate_psnr(pred_lf, lf)
                ssim = calculate_ssim(pred_lf, lf)

                psnr_avg.append(psnr)
                ssim_avg.append(ssim)

                vepoch.set_postfix(PSNR=f"{psnr_avg.get_value():0.4f}({psnr:0.4f})",
                                   SSIM=f"{ssim_avg.get_value():0.4f}({ssim:0.4f})")

            if iters%10 == 0:
                summary_writer.add_scalar('Validation/SSIM', ssim, iters)
                summary_writer.add_scalar('Validation/PSNR', psnr, iters)
                inputs = lf[:, 3, 3, ...]
                inputs = torchvision.utils.make_grid(inputs, nrow=4)
                summary_writer.add_image('Validation/Input', inputs, iters)
                depth = torchvision.utils.make_grid(depth, nrow=4)
                summary_writer.add_image('Validation/Depth', depth, iters)
                N, X, Y, C, H, W = lf.shape
                lf = lf.reshape(N, X*Y, C, H, W)
                for i in range(min(len(lf), 4)):
                    summary_writer.add_video(f'Validation/Target-{i}', lf[i:i+1, ...], iters)
                out = torch.tensor(pred_lf)
                out = out.reshape(N, X*Y, C, H, W)
                for i in range(min(len(lf), 4)):
                    summary_writer.add_video(f'Validation/Output-{i}', out[i:i+1, ...], iters)
            iters += 1


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
    parser.add_argument('-exp', '--exp_name', default='vMPI_trainBoth', type=str, help='name of the experiment')
    parser.add_argument('--results', default='test_results', type=str, help='directory to save results')
    
    parser.add_argument('--gpu', default=1, type=int, help='which gpu to use')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loading')
    
    ######################################## Dataset parameters #######################################
    parser.add_argument('--data_path', default='/media/data/prasan/datasets/LF_datasets', type=str,
                        help='path to dataset')

    parser.add_argument('--filenames_file', default='train_inputs/TAMULF+Stanford/train_files.txt', type=str, 
                        help='path to the filenames text file training')
    parser.add_argument('--filenames_file_eval', default='train_inputs/TAMULF+Stanford/val_files.txt', type=str, 
                        help='path to the filenames text file testing')

    parser.add_argument('-dn', '--depth_network', default='DPT', type=str, 
                        help='depth network used for depth inputs')
    
    parser.add_argument('--unimatch', default=False, action='store_true', help='use unimatch disparity for training')

    ##################################### Learning parameters #########################################
    parser.add_argument('-e', '--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('-bs', '--batchsize', default=8, type=int, help='batch size')
    parser.add_argument('-lr', '--lr', default=2e-5, type=float, help='max learning rate')
    parser.add_argument('-wd', '--wd', default=1e-3, type=float, help='weight decay')
    
    args = parser.parse_args()


    device = torch.device('cuda:{}'.format(args.gpu))
    depth_net = args.depth_network
    if args.unimatch:
        print('Training VMPI with Unimatch')
    else:
        print('Training Variable MPI with {} depth'.format(depth_net))

    feature_extract = True
    train_batch_size = args.batchsize
    val_batch_size = args.batchsize

    num_mpi_planes = 8
    lfsize = [384, 528, 7, 7]
    lfsize_train = [192, 192, 7, 7]
    mode = 'train'
    log_step = 15
    # resume = True

    wandb.login() # env variable WANDB_API_KEY must be set in your environment or manually enter!
    
    logdir = 'logs-TAMULF+Stanford(all_train)/{}-{}'.format(args.exp_name, dt.now().strftime('%d-%h_%H:%M:%S'))
    
    wandb.tensorboard.patch(root_logdir=logdir)
    wandb.init(sync_tensorboard=True, 
                       config=args, 
                       project="lfvr",
                       name=args.exp_name) 
    
    summary_writer = SummaryWriter(logdir)
    os.makedirs('{}/checkpoints'.format(logdir), exist_ok=True)

    # create the model
    model_fix = MpiNet(ngf=32, num_mpi_planes=8, device=device)
    model_fix = model_fix.to(device=device)
    model_vgg = MpiNet(ngf=32, num_mpi_planes=8, device=device)
    model_vgg = model_vgg.to(device=device)

    # create the optimizer
    # stage 1, train the visible network
    #params_to_update = []
    #for name, param in model_fix.named_parameters():
    #    if param.requires_grad == True:
    #        params_to_update.append(param)
    optimizer = optim.Adam(model_fix.parameters(), lr=args.lr, weight_decay=args.wd) # Observe that all parameters are being optimized

    # stage 2, train the occlusion network
    #params_to_update2 = []
    #for name, param in model_vgg.named_parameters():
    #    if param.requires_grad == True:
    #        params_to_update2.append(param)
    optimizer2 = optim.Adam(model_vgg.parameters(), lr=args.lr, weight_decay=args.wd)

    checkpoint = torch.load('checkpoints/visible_net.pth.tar')
    model_fix.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

    checkpoint = torch.load('checkpoints/occluded_net.pth.tar')
    model_vgg.load_state_dict(checkpoint['state_dict'])
    #optimizer2.load_state_dict(checkpoint['optimizer'])

    training_set = LightFieldDataset(args.data_path, args.filenames_file, depth_net, color_corr=True, 
                                     mode='train', unimatch=args.unimatch)
    train_loader = DataLoader(training_set, train_batch_size, shuffle=True, drop_last=True)

    validation_set = LightFieldDataset(args.data_path, args.filenames_file_eval, depth_net, color_corr=True, 
                                       mode='validation', unimatch=args.unimatch)
    val_loader = DataLoader(validation_set, val_batch_size, shuffle=True, drop_last=True)

    repeat = 1
    for i in range(repeat):
        for epoch in range(args.epochs):
            model_fix, optimizer = train_fix_model(model_fix, train_loader, optimizer, epoch+i*args.epochs, repeat*args.epochs)
            save_checkpoint({'state_dict': model_fix.state_dict(), 'optimizer' : optimizer.state_dict()}, filename='{}/checkpoints/visible_{:03d}.pt'.format(logdir, epoch+i*args.epochs))
            evaluate_model(model_vgg, model_fix, val_loader, epoch+i*args.epochs, repeat*args.epochs)

        for epoch in range(args.epochs):
            model_vgg, optimizer2 = train_vgg_model(model_vgg, model_fix, train_loader, optimizer2, epoch+i*args.epochs, repeat*args.epochs)
            save_checkpoint({'state_dict': model_vgg.state_dict(), 'optimizer' : optimizer2.state_dict()}, filename='{}/checkpoints/occluded_{:03d}.pt'.format(logdir, epoch+i*args.epochs))
            evaluate_model(model_vgg, model_fix, val_loader, epoch+i*args.epochs, repeat*args.epochs)