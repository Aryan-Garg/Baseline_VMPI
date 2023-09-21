import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2


    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return 1 - torch.mean(torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1))


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