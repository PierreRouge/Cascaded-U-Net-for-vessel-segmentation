#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:56:46 2022

@author: rouge
"""


import torch
from torch import nn

from monai.networks.layers.factories import Conv
from monai.networks.blocks import Convolution


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
            kernel_size=kernel_size,
            adn_ordering="NDA",
            conv_only=True,
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


class My_Unet(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(My_Unet, self).__init__()
        
        self.conv1 = DoubleConv(dim, in_channel, features[0], strides[0], kernel_size[0])
        self.conv2 = DoubleConv(dim, features[0], features[1], strides[1], kernel_size[1])
        self.conv3 = DoubleConv(dim, features[1], features[2], strides[2], kernel_size[2])
        self.conv4 = DoubleConv(dim, features[2], features[3], strides[3], kernel_size[3])
        self.conv5 = DoubleConv(dim, features[3], features[4], strides[4], kernel_size[4])
        self.conv6 = DoubleConv(dim, features[4], features[5], strides[5], kernel_size[5])
        
        self.up_1 = Conv_Up(dim, features[5], features[4], strides[5], kernel_size[5])
        self.up_2 = Conv_Up(dim, features[4], features[3], strides[4], kernel_size[4])
        self.up_3 = Conv_Up(dim, features[3], features[2], strides[3], kernel_size[3])
        self.up_4 = Conv_Up(dim, features[2], features[1], strides[2], kernel_size[2])
        self.up_5 = Conv_Up(dim, features[1], features[0], strides[1], kernel_size[1])
        
        self.final_conv_1 = Conv["conv", dim](features[0], 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor):
            
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
            
        x_7 = self.up_1(x6, x5)
        x_8 = self.up_2(x_7, x4)
        x_9 = self.up_3(x_8, x3)
        x_10 = self.up_4(x_9, x2)
        x_11 = self.up_5(x_10, x1)
        x_final = self.final_conv_1(x_11)
            
        return x_final
    
    
class My_Unet_tiny(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(My_Unet_tiny, self).__init__()
        
        self.conv1 = DoubleConv(dim, in_channel, features[0], strides[0], kernel_size[0])
        self.conv2 = DoubleConv(dim, features[0], features[1], strides[1], kernel_size[1])
        self.conv3 = DoubleConv(dim, features[1], features[2], strides[2], kernel_size[2])
        self.conv4 = DoubleConv(dim, features[2], features[3], strides[3], kernel_size[3])
        
        self.up_1 = Conv_Up(dim, features[3], features[2], strides[3], kernel_size[3])
        self.up_2 = Conv_Up(dim, features[2], features[1], strides[2], kernel_size[2])
        self.up_3 = Conv_Up(dim, features[1], features[0], strides[1], kernel_size[1])

        self.final_conv_1 = Conv["conv", dim](features[0], 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor):
            
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)

        x_final = self.final_conv_1(x7)
            
        return x_final
    
    
# U-Net for Deep Distance Transform

class DeepDistanceUnet(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size, K=5):
        super(DeepDistanceUnet, self).__init__()
        
        self.conv1 = DoubleConv(dim, in_channel, features[0], strides[0], kernel_size[0])
        self.conv2 = DoubleConv(dim, features[0], features[1], strides[1], kernel_size[1])
        self.conv3 = DoubleConv(dim, features[1], features[2], strides[2], kernel_size[2])
        self.conv4 = DoubleConv(dim, features[2], features[3], strides[3], kernel_size[3])
        
        self.up_1 = Conv_Up(dim, features[3], features[2], strides[3], kernel_size[3])
        self.up_2 = Conv_Up(dim, features[2], features[1], strides[2], kernel_size[2])
        self.up_3 = Conv_Up(dim, features[1], features[0], strides[1], kernel_size[1])

        # Segmentation Head
        self.final_conv_1 = Conv["conv", dim](features[0], 1, kernel_size=1)
        # Distance Map Head (Quantized)
        self.final_conv_2 = Conv["conv", dim](features[0], K, kernel_size=1)
        
    def forward(self, x: torch.Tensor):
            
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)

        x_final_seg = self.final_conv_1(x7)
        x_final_dtm = self.final_conv_2(x7)
            
        return x_final_seg, x_final_dtm
