#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:56:46 2022

@author: rouge
"""

import sys

import torch
from torch import nn

from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Conv

sys.path.append('..')
from network.unet import My_Unet_tiny


class DoubleConv(nn.Sequential):
    def __init__(self, dim, in_features, out_features, strides, kernel_size):
        
        super().__init__()
        
        conv1 = Convolution(
            spatial_dims=dim,
            in_channels=in_features,
            out_channels=out_features,
            strides=strides,
            kernel_size=kernel_size,
            adn_ordering="NDA",
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
            
        )
        
        conv2 = Convolution(
            spatial_dims=dim,
            in_channels=out_features,
            out_channels=out_features,
            strides=1,
            kernel_size=kernel_size,
            adn_ordering="NDA",
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
        )
        
        self.add_module('conv1', conv1)
        self.add_module('conv2', conv2)
        
        
class Conv_Up(nn.Module):
    def __init__(self, dim, in_features, out_features, strides, kernel_size):
        
        super().__init__()
        
        self.conv_trans = Convolution(
            spatial_dims=dim,
            in_channels=in_features,
            out_channels=out_features,
            strides=strides,
            kernel_size=3,
            adn_ordering="NDA",
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
            
        )
        
        self.conv1 = Convolution(
            spatial_dims=dim,
            in_channels=out_features * 2,
            out_channels=out_features,
            strides=(1, 1, 1),
            kernel_size=kernel_size,
            adn_ordering="NDA",
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
            
        )
        
        self.conv2 = Convolution(
            spatial_dims=dim,
            in_channels=out_features,
            out_channels=out_features,
            strides=(1, 1, 1),
            kernel_size=kernel_size,
            adn_ordering="NDA",
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
        )
                
    def forward(self, x, x_encoder):
        x_0 = self.conv_trans(x)
        x_1 = self.conv1(torch.cat([x_encoder, x_0], dim=1))
        x_2 = self.conv2(x_1)
    
        return x_2

    
class Cascaded_Unet(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(Cascaded_Unet, self).__init__()
        
        self.conv_1_1 = DoubleConv(dim, in_channel, features[0], strides[0], kernel_size[0])
        self.conv_1_2 = DoubleConv(dim, features[0], features[1], strides[1], kernel_size[1])
        self.conv_1_3 = DoubleConv(dim, features[1], features[2], strides[2], kernel_size[2])
        self.conv_1_4 = DoubleConv(dim, features[2], features[3], strides[3], kernel_size[3])
        self.conv_1_5 = DoubleConv(dim, features[3], features[4], strides[4], kernel_size[4])
        self.conv_1_6 = DoubleConv(dim, features[4], features[5], strides[5], kernel_size[5])
        
        self.up_1_1 = Conv_Up(dim, features[5], features[4], strides[5], kernel_size[5])
        self.up_1_2 = Conv_Up(dim, features[4], features[3], strides[4], kernel_size[4])
        self.up_1_3 = Conv_Up(dim, features[3], features[2], strides[3], kernel_size[3])
        self.up_1_4 = Conv_Up(dim, features[2], features[1], strides[2], kernel_size[2])
        self.up_1_5 = Conv_Up(dim, features[1], features[0], strides[1], kernel_size[1])
        
        self.sigmoid = nn.Sigmoid()
        
        self.conv_2_1 = DoubleConv(dim, in_channel + 1, features[0], strides[0], kernel_size[0])
        self.conv_2_2 = DoubleConv(dim, features[0], features[1], strides[1], kernel_size[1])
        self.conv_2_3 = DoubleConv(dim, features[1], features[2], strides[2], kernel_size[2])
        self.conv_2_4 = DoubleConv(dim, features[2], features[3], strides[3], kernel_size[3])
        self.conv_2_5 = DoubleConv(dim, features[3], features[4], strides[4], kernel_size[4])
        self.conv_2_6 = DoubleConv(dim, features[4], features[5], strides[5], kernel_size[5])
        
        self.up_2_1 = Conv_Up(dim, features[5], features[4], strides[5], kernel_size[5])
        self.up_2_2 = Conv_Up(dim, features[4], features[3], strides[4], kernel_size[4])
        self.up_2_3 = Conv_Up(dim, features[3], features[2], strides[3], kernel_size[3])
        self.up_2_4 = Conv_Up(dim, features[2], features[1], strides[2], kernel_size[2])
        self.up_2_5 = Conv_Up(dim, features[1], features[0], strides[1], kernel_size[1])
        
        self.final_conv_1 = Conv["conv", dim](features[0], 1, kernel_size=1)
        self.final_conv_2 = Conv["conv", dim](features[0], 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor):
            
        x1 = self.conv_1_1(x)
        x2 = self.conv_1_2(x1)
        x3 = self.conv_1_3(x2)
        x4 = self.conv_1_4(x3)
        x5 = self.conv_1_5(x4)
        x6 = self.conv_1_6(x5)
            
        x_7 = self.up_1_1(x6, x5)
        x_8 = self.up_1_2(x_7, x4)
        x_9 = self.up_1_3(x_8, x3)
        x_10 = self.up_1_4(x_9, x2)
        x_11 = self.up_1_5(x_10, x1)
        x_final_seg = self.final_conv_1(x_11)
        
        x12 = torch.cat((self.sigmoid(x_final_seg), x), dim=1)
        
        x13 = self.conv_2_1(x12)
        x14 = self.conv_2_2(x13)
        x15 = self.conv_2_3(x14)
        x16 = self.conv_2_4(x15)
        x17 = self.conv_2_5(x16)
        x18 = self.conv_2_6(x17)
            
        x19 = self.up_2_1(x18, x17)
        x20 = self.up_2_2(x19, x16)
        x21 = self.up_2_3(x20, x15)
        x22 = self.up_2_4(x21, x14)
        x23 = self.up_2_5(x22, x13)
        x_final_skel = self.final_conv_2(x23)
            
        return x_final_seg, x_final_skel


class Cascaded_Unet_tiny(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(Cascaded_Unet_tiny, self).__init__()
        
        self.conv_1_1 = DoubleConv(dim, in_channel, features[0], strides[0], kernel_size[0])
        self.conv_1_2 = DoubleConv(dim, features[0], features[1], strides[1], kernel_size[1])
        self.conv_1_3 = DoubleConv(dim, features[1], features[2], strides[2], kernel_size[2])
        self.conv_1_4 = DoubleConv(dim, features[2], features[3], strides[3], kernel_size[3])

        self.up_1_1 = Conv_Up(dim, features[3], features[2], strides[3], kernel_size[3])
        self.up_1_2 = Conv_Up(dim, features[2], features[1], strides[2], kernel_size[2])
        self.up_1_3 = Conv_Up(dim, features[1], features[0], strides[1], kernel_size[1])

        self.sigmoid = nn.Sigmoid()
        
        self.conv_2_1 = DoubleConv(dim, in_channel + 1, features[0], strides[0], kernel_size[0])
        self.conv_2_2 = DoubleConv(dim, features[0], features[1], strides[1], kernel_size[1])
        self.conv_2_3 = DoubleConv(dim, features[1], features[2], strides[2], kernel_size[2])
        self.conv_2_4 = DoubleConv(dim, features[2], features[3], strides[3], kernel_size[3])

        self.up_2_1 = Conv_Up(dim, features[3], features[2], strides[3], kernel_size[3])
        self.up_2_2 = Conv_Up(dim, features[2], features[1], strides[2], kernel_size[2])
        self.up_2_3 = Conv_Up(dim, features[1], features[0], strides[1], kernel_size[1])

        self.final_conv_1 = Conv["conv", dim](features[0], 1, kernel_size=1)
        self.final_conv_2 = Conv["conv", dim](features[0], 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor):
            
        x1 = self.conv_1_1(x)
        x2 = self.conv_1_2(x1)
        x3 = self.conv_1_3(x2)
        x4 = self.conv_1_4(x3)

        x5 = self.up_1_1(x4, x3)
        x6 = self.up_1_2(x5, x2)
        x7 = self.up_1_3(x6, x1)

        x_final_seg = self.final_conv_1(x7)
        
        x8 = torch.cat((self.sigmoid(x_final_seg), x), dim=1)
        
        x9 = self.conv_2_1(x8)
        x10 = self.conv_2_2(x9)
        x11 = self.conv_2_3(x10)
        x12 = self.conv_2_4(x11)

        x13 = self.up_2_1(x12, x11)
        x14 = self.up_2_2(x13, x10)
        x15 = self.up_2_3(x14, x9)

        x_final_skel = self.final_conv_2(x15)
            
        return x_final_seg, x_final_skel
    
  
class Cascaded_Unet_tiny_pretrained(nn.Module):
    def __init__(self, dim, in_channel1, in_channel2, features, strides, kernel_size, weights_segmentation, weights_skeleton, freeze_skeleton):
        super(Cascaded_Unet_tiny_pretrained, self).__init__()
        
        self.network_segmentation = My_Unet_tiny(dim, in_channel1, features, strides, kernel_size)
        self.network_skeleton = My_Unet_tiny(dim, in_channel2, features, strides, kernel_size)
        
        self.network_segmentation.load_state_dict(weights_segmentation)
        self.network_skeleton.load_state_dict(weights_skeleton)
        
        if freeze_skeleton:
            for p in self.network_skeleton.parameters():
                p.requires_grad = False
            
    def forward(self, x: torch.Tensor):
            
        x_segmentation = self.network_segmentation(x)
        x_skel = self.network_skeleton(torch.cat((x, x_segmentation), dim=1))
            
        return x_segmentation, x_skel
