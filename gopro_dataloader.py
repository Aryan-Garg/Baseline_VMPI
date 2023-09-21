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


lfsize = [352, 528]
lfsize_train = [256, 256]
lfsize_test = [352, 528]
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


# dataset
class LightFieldDataset(Dataset):
    def __init__(self, path, filenames_file, depth_network, color_corr=False, transform=None, mode='train'):
        self.data_path = path
        self.depth_path = os.path.join(path, '{}-depth'.format(depth_network))
        self.mode = mode
        self.color_corr = color_corr
        with open(filenames_file, 'r') as f:
            self.filenames = f.readlines()
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        W, H = 528, 352

        file_path = self.filenames[idx][:-1]
        img = Image.open(os.path.join(self.data_path, file_path))
        img  = img.resize((W, H))
        img = np.array(img)/255.0
        img = normalize_lf(img)
        img = torch.tensor(img, dtype=torch.float)
        #img  = img.permute([2, 0, 1])
        
        # get deeplens depth
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

        return img, depth
