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
from PIL import Image
import skimage
import skimage.transform
from skimage.exposure import adjust_gamma


lfsize = [480, 640, 7, 7]
lfsize_train = [256, 256, 7, 7]
# lfsize_test = [352, 528, 7, 7]
T = np.load('color_transfer.npy')

def normalize_lf(lf):
    return 2.0*(lf-0.5)


def denormalize_lf(lf):
    return lf/2.0+0.5


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


def color_transform(src, T):
    """
    src: [H, W, 3]
    T: [3, 256]
    """
    res = np.zeros_like(src, dtype=np.float)
    for i in range(3):
        for j in range(src.shape[0]):
            for k in range(src.shape[1]):
                res[j][k][i] = T[i,src[j,k,i]]*1.0
                if res[j,k,i] < 0:
                    res[j,k,i] = 0
                if res[j,k,i] > 1:
                    res[j,k,i] = 1
    res = res*255.0
    return res.astype(np.uint8)


def data_augmentation(x, y, device, mode='train'):
    """
    x, y: np array, [B, C, H, W]
    """
    x_list = list()
    y_list = list()
    for i in range(x.shape[0]):
        xi = (255*x[i, ...]).astype(np.uint8)
        yi = (255*y[i, ...]).astype(np.uint8)
        xi = np.transpose(xi, (1, 2, 0))
        yi = np.transpose(yi, (1, 2, 0))

        # gamma correction
        #xi = adjust_gamma(xi, 0.8)
        #yi = adjust_gamma(yi, 0.8)

        # color transfer
        #if mode == 'train':
        #    xi = color_transform(xi, T)
        #    yi = color_transform(yi, T)
        
        xi = Image.fromarray(xi)
        yi = Image.fromarray(yi)

        if mode == 'train':
            pick = random.randint(0, 4)
            if pick == 0:
                # random brightness
                brightness_factor = 1.0 + random.uniform(0, 0.3)
                xi = ttf.adjust_brightness(xi, brightness_factor)
                yi = ttf.adjust_brightness(yi, brightness_factor)
            elif pick == 1:
                # random saturation
                saturation_factor = 1.0 + random.uniform(-0.2, 0.5)
                xi = ttf.adjust_saturation(xi, saturation_factor)
                yi = ttf.adjust_saturation(yi, saturation_factor)
            elif pick == 2:
                # random hue
                hue_factor = random.uniform(-0.2, 0.2)
                xi = ttf.adjust_hue(xi, hue_factor)
                yi = ttf.adjust_hue(yi, hue_factor)
            elif pick == 3:
                # random contrast
                contrast_factor = 1.0 + random.uniform(-0.2, 0.4)
                xi = ttf.adjust_contrast(xi, contrast_factor)
                yi = ttf.adjust_contrast(yi, contrast_factor)
            elif pick == 4:
                # random swap color channel
                permute = np.random.permutation(3)
                xi = np.array(xi)
                yi = np.array(yi)
                xi = xi[..., permute]
                yi = yi[..., permute]
        
        xi = np.clip(np.array(xi)/255.0, 0, 1.0)
        yi = np.clip(np.array(yi)/255.0, 0, 1.0)
        x_list.append(xi)
        y_list.append(yi)
    x_ret = torch.tensor(np.stack(x_list, axis=0), dtype=torch.float)
    y_ret = torch.tensor(np.stack(y_list, axis=0), dtype=torch.float)
    x_ret = normalize_lf(x_ret)
    y_ret = normalize_lf(y_ret)

    return x_ret.to(device), y_ret.to(device)



# dataset
class LightFieldDataset(Dataset):
    def __init__(self, path, filenames_file, depth_network, color_corr=False, transform=None, mode='train', unimatch=False, lfsize_test=None):
        self.unimatch = unimatch
        self.lfsize_test = lfsize_test
        self.lf_path = path
        if unimatch:
            self.depth_path = "/data2/aryan/unimatch/dp_otherDS/"
        else:
            self.depth_path = os.path.join(path, '{}-depth'.format(depth_network))
        self.mode = mode
        self.color_corr = color_corr
        with open(filenames_file, 'r') as f:
            self.filenames = f.readlines()
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = self.filenames[idx][:-1]
        lf = np.load(os.path.join(self.lf_path, file_path))/255.0
        
        if self.color_corr:
            mean = lf.mean()
            fact = np.log(0.4) / np.log(mean)
            if fact<1:
                lf = lf ** fact

        lf = torch.tensor(lf, dtype=torch.float)
        if lf.shape[4] == 3:
            lf = lf.permute([0, 1, 4, 2, 3])
        else:
            lf = lf.permute([1, 0, 2, 3, 4])
        #print(lf.shape)
        V, V, C, H, W = lf.shape
        lf = lf.reshape((V*V, C, H, W))
        lf = F.interpolate(lf, size=lfsize[:2], mode='nearest')
        H, W = lfsize[:2]
        lf = lf.reshape(V, V, C, H, W)
        
        aif = lf

        # get deeplens depth
        file_path = file_path.replace('npy', 'png')
        if self.unimatch:
            file_name = file_path.split("/")[-1]
            file_name = "left_" + file_name.split(".")[0] + "_disp.png"
            depth = Image.open(os.path.join(self.depth_path, file_path.split("/")[0] + "/" + file_path.split("/")[1] + "/" + file_name)).convert('L')
        else:
            depth = Image.open(os.path.join(self.depth_path, file_path)).convert('L')

        depth = depth.resize((W, H))
        depth = np.array(depth)/255.0
        
        depth = np.expand_dims(depth, axis=0)
        if 'DPT' in self.depth_path:
            depth = depth
            #depth = 1 / (depth + 1e-2)
            mini, maxi = depth.min(), depth.max()
            depth = (depth - mini) / (maxi - mini)
        depth = torch.tensor(depth, dtype=torch.float)
        #depth = 1 - depth

        if self.mode == 'train':
            height, width = lfsize_train[:2]
            x = random.randint(0, lf.shape[-1] - width)
            y = random.randint(0, lf.shape[-2] - height)
            lf = lf[:, :, :, y:y + height, x:x + width]
            depth = depth[:, y:y + height, x:x + width]
            
        elif self.mode == 'validation':
            height, width = self.lfsize_test[:2]
            x = random.randint(0, lf.shape[-1] - width)
            y = random.randint(0, lf.shape[-2] - height)
            lf = lf[:, :, :, y:y + height, x:x + width]
            depth = depth[:, y:y + height, x:x + width]

        return aif, lf, depth
