import os
import random
import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import mpi

num_mpi_planes = 8
Run_stats = True 
AFF = True  # stop updating batch norm layers
MOM = 0.0

class MpiNet(nn.Module):
    def __init__(self, ngf=32, num_mpi_planes=8, device=torch.device('cuda:0')):
        super(MpiNet, self).__init__()

        self.num_outputs = num_mpi_planes*5
        self.num_mpi_planes = num_mpi_planes
        self.device = device

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
        self.conv8_3 = nn.Conv2d(ngf*2, self.num_outputs, 1)

    def forward(self, x, depth, v, u):
        # deeplens depth
        depth_features = depth.permute([0, 1, 2, 3])
        x = x.permute([0, 3, 1, 2])

        #print(x.shape, depth_features.shape)
        # get image planes
        c1_1 = self.conv1_1(torch.cat((x, depth_features), 1))
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

        N, _, H, W = c8_3.shape
        rgba_layers = torch.reshape(c8_3.permute([0, 2, 3, 1]), (N, H, W, self.num_mpi_planes, 5))
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
            output.append(mpi.mpi_lf_rendering(rgba_layers[i:i+1,...], depth_planes[i], v, u, self.device))
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
