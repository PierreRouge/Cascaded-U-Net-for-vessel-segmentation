#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:44:04 2024

@author: rouge
3D Channel and Spatial Attention Network (CSA-Net 3D). Copied from https://github.com/iMED-Lab/CS-Net/blob/master/model/csnet_3d.py (Mou et al. Media 2020)
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # print("Is nan x")
        # print(torch.any(torch.isnan(x)))
        residual = self.conv1x1(x)
        # print("Is nan residual")
        # print(torch.any(torch.isnan(residual)))
        out = self.relu(self.bn1(self.conv1(x)))
        # print("Is nan out")
        # print(torch.any(torch.isnan(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        # print("Is nan out")
        # print(torch.any(torch.isnan(out)))
        out = out + residual
        # print("Is nan out")
        # print(torch.any(torch.isnan(out)))
        out = self.relu(out)
        # print("Is nan out")
        # print(torch.any(torch.isnan(out)))
        return out


class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SpatialAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock3d, self).__init__()
        self.query = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            nn.BatchNorm3d(in_channels // 8),
            nn.ReLU(inplace=False)
        )
        self.key = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(in_channels // 8),
            nn.ReLU(inplace=False)
        )
        self.judge = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(in_channels // 8),
            nn.ReLU(inplace=False)
        )
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxZ )
        :return: affinity value + x
        B: batch size
        C: channels
        H: height
        W: width
        D: slice number (depth)
        """
        # print('max')
        # print(torch.max(x))
        B, C, H, W, D = x.size()
        # compress x: [B,C,H,W,Z]-->[B,H*W*Z,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,W*H*D,C]
        proj_key = self.key(x).view(B, -1, W * H * D)  # -> [B,H*W*D,C]
        proj_judge = self.judge(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,C,H*W*D]
        
        # print('max')
        # print(torch.max(proj_query))
        # print('max')
        # print(torch.max(proj_key))
        # print('max')
        # print(torch.max(proj_judge))
        
        affinity1 = torch.matmul(proj_query, proj_key)
        # print('max')
        # print(torch.max(affinity1))
        affinity2 = torch.matmul(proj_judge, proj_key)
        # print('max')
        # print(torch.max(affinity2))
        affinity = torch.matmul(affinity1, affinity2)
        # print("Is nan affinity 1")
        # print(torch.any(torch.isnan(affinity)))
        # print('max')
        # print(torch.max(affinity))
        # print('min')
        # print(torch.min(affinity))
        affinity = self.softmax(affinity)
        # print("Is nan affinity 2")
        # print(torch.any(torch.isnan(affinity)))

        proj_value = self.value(x).view(B, -1, H * W * D)  # -> C*N
        weights = torch.matmul(proj_value, affinity)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock3d, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxD )
        :return: affinity value + x
        """
        B, C, H, W, D = x.size()
        proj_query = x.view(B, C, -1).permute(0, 2, 1)
        proj_key = x.view(B, C, -1)
        proj_judge = x.view(B, C, -1).permute(0, 2, 1)
        affinity1 = torch.matmul(proj_key, proj_query)
        affinity2 = torch.matmul(proj_key, proj_judge)
        affinity = torch.matmul(affinity1, affinity2)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        # print("Is nan affinity_new 1")
        # print(torch.any(torch.isnan(affinity_new)))
        affinity_new = self.softmax(affinity_new)
        # print("Is nan affinity_new 2")
        # print(torch.any(torch.isnan(affinity_new)))
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out


class AffinityAttention3d(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention3d, self).__init__()
        self.sab = SpatialAttentionBlock3d(in_channels)
        self.cab = ChannelAttentionBlock3d(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        # print("Is nan x")
        # print(torch.any(torch.isnan(x)))
        sab = self.sab(x)
        # print("Is nan sab")
        # print(torch.any(torch.isnan(sab)))
        
        cab = self.cab(x)
        # print("Is nan cab")
        # print(torch.any(torch.isnan(cab)))
        
        out = sab + cab + x
        
        # print("Is nan out")
        # print(torch.any(torch.isnan(out)))
        return out


class CSNet3D(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(CSNet3D, self).__init__()
        self.enc_input = ResEncoder3d(channels, 16)
        self.encoder1 = ResEncoder3d(16, 32)
        self.encoder2 = ResEncoder3d(32, 64)
        self.encoder3 = ResEncoder3d(64, 128)
        self.encoder4 = ResEncoder3d(128, 256)
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention3d(256)
        # self.attention_fuse = nn.Conv3d(256 * 2, 256, kernel_size=1)
        self.decoder4 = Decoder3d(256, 128)
        self.decoder3 = Decoder3d(128, 64)
        self.decoder2 = Decoder3d(64, 32)
        self.decoder1 = Decoder3d(32, 16)
        self.deconv4 = deconv(256, 128)
        self.deconv3 = deconv(128, 64)
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(32, 16)
        self.final = nn.Conv3d(16, classes, kernel_size=1)
        # initialize_weights(self)

    def forward(self, x):
        # print("Is nan x")
        # print(torch.any(torch.isnan(x)))
        enc_input = self.enc_input(x)
        # print("Is nan enc input")
        # print(torch.any(torch.isnan(enc_input)))
        down1 = self.downsample(enc_input)
        # print("Is nan down1 input")
        # print(torch.any(torch.isnan(down1)))

        enc1 = self.encoder1(down1)
        # print("Is nan enc1")
        # print(torch.any(torch.isnan(enc1)))
        down2 = self.downsample(enc1)
        # print("Is nan down2")
        # print(torch.any(torch.isnan(down2)))

        enc2 = self.encoder2(down2)
        # print("Is nan enc2")
        # print(torch.any(torch.isnan(enc2)))
        down3 = self.downsample(enc2)
        # print("Is nan down3")
        # print(torch.any(torch.isnan(down3)))

        enc3 = self.encoder3(down3)
        # print("Is nan enc3")
        # print(torch.any(torch.isnan(enc3)))
        down4 = self.downsample(enc3)
        # print("Is nan down4")
        # print(torch.any(torch.isnan(down4)))

        input_feature = self.encoder4(down4)
        # print("Is nan input_feature")
        # print(torch.any(torch.isnan(input_feature)))

        # Do Attenttion operations here
        attention = self.affinity_attention(input_feature)
        # print("Is nan attention")
        # print(torch.any(torch.isnan(attention)))
        # attention_fuse = input_feature + attention
        # print("Is nan attention fuse")
        # print(torch.any(torch.isnan(attention_fuse)))

        # Do decoder operations here
        up4 = self.deconv4(attention)
        # print("Is nan up4")
        # print(torch.any(torch.isnan(up4)))
        up4 = torch.cat((enc3, up4), dim=1)
        # print("Is nan up4")
        # print(torch.any(torch.isnan(up4)))
        dec4 = self.decoder4(up4)
        # print("Is nan dec4")
        # print(torch.any(torch.isnan(dec4)))

        up3 = self.deconv3(dec4)
        # print("Is nan up3")
        # print(torch.any(torch.isnan(up3)))
        up3 = torch.cat((enc2, up3), dim=1)
        # print("Is nan up3")
        # print(torch.any(torch.isnan(up3)))
        dec3 = self.decoder3(up3)
        # print("Is nan dec3")
        # print(torch.any(torch.isnan(dec3)))

        up2 = self.deconv2(dec3)
        # print("Is nan up2")
        # print(torch.any(torch.isnan(up2)))
        up2 = torch.cat((enc1, up2), dim=1)
        # print("Is nan up2")
        # print(torch.any(torch.isnan(up2)))
        dec2 = self.decoder2(up2)
        # print("Is nan dec2")
        # print(torch.any(torch.isnan(dec2)))

        up1 = self.deconv1(dec2)
        # print("Is nan up1")
        # print(torch.any(torch.isnan(up1)))
        up1 = torch.cat((enc_input, up1), dim=1)
        # print("Is nan up1")
        # print(torch.any(torch.isnan(up1)))
        dec1 = self.decoder1(up1)
        # print("Is nan dec1")
        # print(torch.any(torch.isnan(dec1)))

        final = self.final(dec1)
        # print("Is nan final")
        # print(torch.any(torch.isnan(final)))
        return final
