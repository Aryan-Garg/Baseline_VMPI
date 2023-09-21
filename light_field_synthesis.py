# Input Single 

import os
import random
import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as ttf
import torch.optim as optim
from PIL import Image
import skimage
import skimage.transform
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import mpi
from vgg import vgg_loss, weighted_vgg_loss
from loss import SSIMLoss


device = torch.device('cuda:0')
feature_extract = True
batch_size = 1
train_batch_size = 8
num_mpi_planes = 8
lfsize = [372, 540, 7, 7]
lfsize_train = [256, 256, 7, 7]
lfsize_test = [352, 512, 7, 7]
Run_stats = True 
AFF = True  # stop updating batch norm layers
MOM = 0.0
mode = 'validate'
#resume = True


def normalize_lf(lf):
    return 2.0*(lf-0.5)

def denormalize_lf(lf):
    return lf/2.0+0.5

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

class MpiNet(nn.Module):
    def __init__(self, ngf=32, num_outputs=num_mpi_planes*5):
        super(MpiNet, self).__init__()

        # rgba network
        self.conv1_1 = nn.Conv2d(4, ngf, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(ngf, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv1_2 = nn.Conv2d(ngf, ngf*2, 3, padding=1, stride=2)
        self.bn1_2 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv2_1 = nn.Conv2d(ngf*2, ngf*2, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv2_2 = nn.Conv2d(ngf*2, ngf*4, 3, padding=1, stride=2)
        self.bn2_2 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv3_1 = nn.Conv2d(ngf*4, ngf*4, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv3_2 = nn.Conv2d(ngf*4, ngf*4, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv3_3 = nn.Conv2d(ngf*4, ngf*8, 3, padding=1, stride=2)
        self.bn3_3 = nn.BatchNorm2d(ngf*8, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv4_1 = nn.Conv2d(ngf*8, ngf*8, 3, padding=2, dilation=2)
        self.bn4_1 = nn.BatchNorm2d(ngf*8, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv4_2 = nn.Conv2d(ngf*8, ngf*8, 3, padding=2, dilation=2)
        self.bn4_2 = nn.BatchNorm2d(ngf*8, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv4_3 = nn.Conv2d(ngf*8, ngf*8, 3, padding=2, dilation=2)
        self.bn4_3 = nn.BatchNorm2d(ngf*8, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv6_1 = nn.ConvTranspose2d(ngf*16, ngf*4, 4, padding=1, stride=2)
        self.bn6_1 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv6_2 = nn.Conv2d(ngf*4, ngf*4, 3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv6_3 = nn.Conv2d(ngf*4, ngf*4, 3, padding=1)
        self.bn6_3 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv7_1 = nn.ConvTranspose2d(ngf*8, ngf*2, 4, padding=1, stride=2)
        self.bn7_1 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv7_2 = nn.Conv2d(ngf*2, ngf*2, 3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv8_1 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, padding=1, stride=2)
        self.bn8_1 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv8_2 = nn.Conv2d(ngf*2, ngf*2, 3, padding=1)
        self.bn8_2 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv8_3 = nn.Conv2d(ngf*2, num_outputs, 1)

    def forward(self, x, depth, v, u):
        # deeplens depth
        depth_features = depth.permute([0, 3, 1, 2])
        #x = x.permute([0, 3, 1, 2])
        #print(x.shape, depth_features.shape, x.type(), depth_features.type())

        # get image planes
        inp = torch.cat((x, depth_features), dim=1)
        c1_1 = self.conv1_1(inp)
        c1_1 = self.bn1_1(c1_1)
        c1_1 = F.relu(c1_1)
        c1_2 = self.conv1_2(c1_1)
        c1_2 = self.bn1_2(c1_2)
        c1_2 = F.relu(c1_2)
        c2_1 = self.conv2_1(c1_2)
        c2_1 = self.bn2_1(c2_1)
        c2_1 = F.relu(c2_1)
        c2_2 = self.conv2_2(c2_1)
        c2_2 = self.bn2_2(c2_2)
        c2_2 = F.relu(c2_2)
        c3_1 = self.conv3_1(c2_2)
        c3_1 = self.bn3_1(c3_1)
        c3_1 = F.relu(c3_1)
        c3_2 = self.conv3_2(c3_1)
        c3_2 = self.bn3_2(c3_2)
        c3_2 = F.relu(c3_2)
        c3_3 = self.conv3_3(c3_2)
        c3_3 = self.bn3_3(c3_3)
        c3_3 = F.relu(c3_3)
        c4_1 = self.conv4_1(c3_3)
        c4_1 = self.bn4_1(c4_1)
        c4_1 = F.relu(c4_1)
        c4_2 = self.conv4_2(c4_1)
        c4_2 = self.bn4_2(c4_2)
        c4_2 = F.relu(c4_2)
        c4_3 = self.conv4_3(c4_2)
        c4_3 = self.bn4_3(c4_3)
        c4_3 = F.relu(c4_3)
        
        c6_1 = self.conv6_1(torch.cat((c4_3, c3_3), 1))
        c6_1 = self.bn6_1(c6_1)
        c6_1 = F.relu(c6_1)
        c6_2 = self.conv6_2(c6_1)
        c6_2 = self.bn6_2(c6_2)
        c6_2 = F.relu(c6_2)
        c6_3 = self.conv6_3(c6_2)
        c6_3 = self.bn6_3(c6_3)
        c6_3 = F.relu(c6_3)
        c7_1 = self.conv7_1(torch.cat((c6_3, c2_2), 1))
        c7_1 = self.bn7_1(c7_1)
        c7_1 = F.relu(c7_1)
        c7_2 = self.conv7_2(c7_1)
        c7_2 = self.bn7_2(c7_2)
        c7_2 = F.relu(c7_2)
        c8_1 = self.conv8_1(torch.cat((c7_2, c1_2), 1))
        c8_1 = self.bn8_1(c8_1)
        c8_1 = F.relu(c8_1)
        c8_2 = self.conv8_2(c8_1)
        c8_2 = self.bn8_2(c8_2)
        c8_2 = F.relu(c8_2)
        c8_3 = self.conv8_3(c8_2)
        c8_3 = torch.tanh(c8_3)

        rgba_layers = torch.reshape(c8_3.permute([0, 2, 3, 1]), (batch_size, lfsize[0], lfsize[1], num_mpi_planes, 5))
        color_layers = rgba_layers[:, :, :, :, :3]
        alpha_layers = rgba_layers[:, :, :, :, 3:4]
        depth_layers = rgba_layers[:, :, :, :, 4]
        # Rescale alphas to (0, 1)
        alpha_layers = (alpha_layers + 1.) / 2.
        rgba_layers = torch.cat((color_layers, alpha_layers), 4)
        rgbad_layers = torch.cat((color_layers, alpha_layers, torch.unsqueeze(depth_layers, dim=-1)), 4)

        depth_planes = depth_layers.mean(1).mean(1)

        # rendering
        output = list()
        for i in range(rgba_layers.shape[0]):
            output.append(mpi.mpi_lf_rendering(rgba_layers[i:i+1,...], depth_planes[i], v, u))
        output = torch.cat(output, dim=0)

        return output, rgbad_layers

    def load_network(self, filename):
        model = torch.load(filename)
        return model
    
    def save_network(self, network, model_path):
        torch.save(network.cpu().state_dict(), model_path)
        model.to(device=device)
        return True

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

def lock_mpi_net(model):
    for name, param in model.named_parameters():
        if param.requires_grad and 'inp' not in name:
            param.requires_grad = False


# dataset
class LightFieldDataset(Dataset):
    def __init__(self, path, filenames_file, transform=None, mode='train'):
        self.lf_path = path
        self.depth_path = os.path.join(path, 'DPT-depth')
        self.mode = mode
        with open(filenames_file, 'r') as f:
            self.filenames = f.readlines()
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = self.filenames[idx][:-1]
        lf = np.load(os.path.join(self.lf_path, file_path)).astype(np.float32)
        img = lf[3, 3, ...]#(255 * lf[3, 3, ...]).astype(np.uint8)
        img - normalize_lf(img)
        img = torch.tensor(img, dtype=torch.float)
        #lf = Image.open(self.filenames[idx])
        
        aif = lf

        # get deeplens depth
        file_path = file_path.replace('npy', 'png')
        depth = Image.open(os.path.join(self.depth_path, file_path)).convert('L')
        depth = np.asarray(depth)/255.0
        
        depth = np.expand_dims(depth, axis=-1)
        depth = torch.tensor(depth, dtype=torch.float)
        depth = 1 / (depth + 1e-2)

        maxi = depth.max()
        mini = depth.min()
        depth = (depth - mini)/(maxi - mini)
        
        return aif, img, lf, depth


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

    return weight

def lf_loss(lf_shear, labels, v, u, rgbad):
    """
    Args:
        lf_shear: [B, H, W, C]
        labels: [B, H, W, C]
    """
    shear_loss = torch.mean(torch.abs(denormalize_lf(lf_shear) - denormalize_lf(labels)))
    #beta = 0.005
    #vgg_loss = beta*lf_weighted_loss(lf_shear, labels, v, u, rgbad_init)

    return shear_loss + depth_loss #+ vgg_loss

def lf_loss_l1(lf_shear, labels):
    """
    Args:
        lf_shear: [B, H, W, C]
        labels: [B, H, W, C]
    """
    shear_loss = torch.mean(torch.abs(denormalize_lf(lf_shear) - denormalize_lf(labels)))
    return shear_loss


def save_checkpoint(state, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)

def change_param_to_eval():
    global lfsize, batch_size
    lfsize[0] = lfsize_test[0]
    lfsize[1] = lfsize_test[1]
    batch_size = 1

def change_param_to_train():
    global lfsize, batch_size
    lfsize[0] = lfsize_train[0]
    lfsize[1] = lfsize_train[1]
    batch_size = train_batch_size


def calculate_psnr(img1, img2):
    with torch.no_grad():
        img1 = 255 * img1
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


def evaluate_model(model_vgg, model_init, val_loader):
    model_init.eval()
    model_vgg.eval()
    change_param_to_eval()
    total_psnr = 0.0
    total_ssim = 0.0

    for i_batch, batched_inputs in enumerate(val_loader):
        img = batched_inputs[1].to(device)
        lf = batched_inputs[2].numpy()/255.
        depth = batched_inputs[3].to(device)
        #print(img.shape, depth.shape, lf.shape)
        print(img.max(), lf.max())

        with torch.set_grad_enabled(False):
            pred_lf = np.zeros(lf.shape)

            for v in range(0, 7):
                for u in range(0, 7):
                    #print(u, v)
                    inputs = img
                    target = img
                    #inputs, target = data_augmentation(inputs.numpy(), target.numpy(), mode='val')
                    inputs = normalize_lf(inputs/255.0)

                    lf_shear_init, rgbad_init = model_fix(inputs, depth, v-3, u-3)
                    mask = get_mask(rgbad_init.detach(), v-3, u-3)
                    mask_rest = 1.0 - mask
                    lf_shear2, rgbad2 = model_vgg(inputs, depth.to(device), v-3, u-3)
                    final_output = mask_rest*lf_shear_init + mask*lf_shear2
                    final_output = denormalize_lf(final_output.permute(0, 3, 1, 2))
                    print(final_output.max())

                    pred_lf[:,v,u,:,:,:] = final_output.cpu().numpy()
                    
            psnr = calculate_psnr(pred_lf, lf)
            ssim = calculate_ssim(pred_lf, lf)

            total_psnr += psnr
            total_ssim += ssim

            #np.save('results_original/output5_%d.npy'%(i_batch), denormalize_lf(lf))
            #np.save('results/output10_%d.npy'%(i_batch), lf)
            #np.save('results/rgbad_%d.npy'%(i_batch), rgbad_init.cpu().numpy())
            print('Average PSNR %f, Average SSIM: %lf' %(total_psnr/len(val_loader), total_ssim/(i_batch+1)))


# create the model
model_vgg = MpiNet()
model_fix = MpiNet()

model_vgg = model_vgg.to(device=device)
model_fix = model_fix.to(device=device)


if __name__ == '__main__':
    validation_set = LightFieldDataset('/media/data/prasan/datasets/LF_datasets', 'tamulf/test_files.txt', mode='validate')
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False)

    checkpoint = torch.load('checkpoints/occluded_net.pth.tar', map_location='cpu')
    model_vgg.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load('checkpoints/visible_net.pth.tar', map_location='cpu')
    model_fix.load_state_dict(checkpoint['state_dict'])
    evaluate_model(model_vgg, model_fix, val_loader)