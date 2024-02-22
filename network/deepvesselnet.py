#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:56:46 2022

@author: rouge
Reimplementation of DeepVesselNet Tetteh et al.
"""


import torch
from torch import nn


class Conv3dCH(nn.Module):
    def __init__(self, in_features, out_features, strides, kernel_size):
        super(Conv3dCH, self).__init__()
        
        self.convx = nn.Conv3d(in_features, out_features, (kernel_size, 1, kernel_size), stride=strides, padding='same')
        self.convy = nn.Conv3d(in_features, out_features, (kernel_size, kernel_size, 1), stride=strides, padding='same')
        self.convz = nn.Conv3d(in_features, out_features, (1, kernel_size, kernel_size), stride=strides, padding='same')

    def forward(self, x):
        cx = self.convx(x)
        cy = self.convy(x)
        cz = self.convz(x)
        out = torch.add(cx, cy)
        out = torch.add(out, cz)
        return out


class BlockConv3dCH(nn.Sequential):
    def __init__(self, dim, in_features, out_features, strides, kernel_size):
        
        super().__init__()
        
        conv1 = nn.Sequential(
            Conv3dCH(in_features, out_features, strides, kernel_size),
            nn.InstanceNorm3d(num_features=out_features, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False)
        )
        
        self.add_module('conv1', conv1)


class DeepVesselNet(nn.Module):
    def __init__(self, dim=3, in_channel=1, features=(5, 10, 20, 50), strides=(1, 1, 1, 1), kernel_size=(3, 5, 5, 3)):
        super(DeepVesselNet, self).__init__()
              
        self.conv1 = BlockConv3dCH(dim=dim, in_features=in_channel, out_features=features[0], strides=strides[0], kernel_size=kernel_size[0])
        self.conv2 = BlockConv3dCH(dim=dim, in_features=features[0], out_features=features[1], strides=strides[1], kernel_size=kernel_size[1])
        self.conv3 = BlockConv3dCH(dim=dim, in_features=features[1], out_features=features[2], strides=strides[2], kernel_size=kernel_size[2])
        self.conv4 = BlockConv3dCH(dim=dim, in_features=features[2], out_features=features[3], strides=strides[3], kernel_size=kernel_size[3])
        
        self.final_conv = Conv3dCH(in_features=features[3], out_features=1, strides=1, kernel_size=1)
        
    def forward(self, x: torch.Tensor):
            
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x_final = self.final_conv(x4)
            
        return x_final
