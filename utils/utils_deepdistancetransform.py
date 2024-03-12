#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:41:38 2024

@author: rouge
"""
import numpy as np
import torch
from torch import nn
import time
import matplotlib.pyplot as plt
import nibabel as nib

from glob import glob

from scipy.ndimage.morphology import distance_transform_edt

import sys
sys.path.append('..')
from utils.utils_pytorch import dice_metric_pytorch, dice_loss_pytorch, sigmoid_clip


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

sigmoid = sigmoid_clip(1E-3)
softmax = nn.Softmax(dim=1)


def distance_transform(binary_array):
    dtm = np.round(distance_transform_edt(binary_array))
    return dtm


def loss_ddt(K):
    def loss(y_pred, y_true):
        y_pred_softmax = softmax(y_pred)
        wv = torch.abs(torch.argmax(y_pred_softmax, dim=1) - y_true) / K
        max_tensor, _ = torch.max(y_pred_softmax, dim=1)
        value = torch.mean(wv * torch.log(1 - max_tensor))
        return value
    return loss


# Training
def train_loop_DeepDistance(dataloader, validloader, model, loss_param, input_, optimizer, device, epoch, max_epoch, K):

    if loss_param == "deep-distance-transform":
        loss_0 = dice_loss_pytorch
        loss_1 = nn.BCELoss()
        
        if K == 8:
            weights = torch.tensor([1 / 25000000, 1 / 100000, 1 / 5000, 1 / 800, 1 / 150, 1 / 50, 1 / 20, 1 / 5], device=device)
            # weights = torch.tensor([1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], device=device)
        elif K == 7:
            weights = torch.tensor([1 / 25000000, 1 / 70000, 1 / 10000, 1 / 1000, 1 / 100, 1 / 10, 1], device=device)
        loss_ce_dtm = nn.CrossEntropyLoss(weight=weights)
        # loss_term = loss_ddt(K)
        
        model.eval()
        val_loss_0 = 0.0
        val_dice_0 = 0.0
        with torch.no_grad():
            for batch, data in enumerate(validloader):
                if input_ == 'MRI':
                    X_val = data['image']
                elif input_ == 'segmentation':
                    X_val = data['segmentation']
                elif input_ == 'Both':
                    X_val_1 = data['image']
                    X_val_2 = data['segmentation']
                    X_val = torch.cat((X_val_1, X_val_2), dim=1)
                
                y_val_seg = data['GT']
                y_val_dtm = data['DTM']
                y_val_seg = y_val_seg.to(device)
                y_val_dtm = y_val_dtm.to(device)
                b, c, s1, s2, s3 = y_val_dtm.shape
                
       
                X_val = X_val.float()
                X_val = X_val.to(device)
                
                pred_val_seg, pred_val_dtm = model(X_val)
                pred_val_seg = sigmoid(pred_val_seg)
                # pred_val_dtm = softmax(pred_val_dtm)
                
                
                print("Unique pred val dtm")
                print(torch.max(softmax(pred_val_dtm)))
                print("Unique pred val dtm")
                print(torch.unique(torch.argmax(softmax(pred_val_dtm), dim=1)))
                print("Unique y dtm val")
                print(torch.unique(y_val_dtm.long()))
                
                
                y_val_dtm = torch.clamp(y_val_dtm.view(b, s1, s2, s3).long(), min=0, max=K - 1)
                val_loss = loss_0(pred_val_seg, y_val_seg) + loss_1(pred_val_seg, y_val_seg) + loss_ce_dtm(pred_val_dtm, y_val_dtm) #+ loss_term(pred_val_dtm, y_val_dtm)
                val_loss = val_loss.item()
                val_loss_0 += val_loss
                
                pred_val_seg = nn.functional.threshold(pred_val_seg, threshold=0.5, value=0)
                ones = torch.ones(pred_val_seg.shape, dtype=torch.float)
                ones = ones.to(device)
                pred_val_seg = torch.where(pred_val_seg > 0, ones, pred_val_seg)
                dice_val = dice_metric_pytorch(pred_val_seg, y_val_seg)
                dice_val = dice_val.cpu().detach().numpy()
                dice_val = np.mean(dice_val)
                val_dice_0 += dice_val.item()
    
            val_loss = val_loss_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            
        train_loss = 0.0
        train_loss_dice = 0.0
        train_loss_bce = 0.0
        train_loss_dtm = 0.0
        start = time.time()
        for batch_train, data in enumerate(dataloader):
            if input_ == 'MRI':
                X = data['image']
            elif input_ == 'segmentation':
                X = data['segmentation']
            elif input_ == 'Both':
                X_1 = data['image']
                X_2 = data['segmentation']
                X = torch.cat((X_1, X_2), dim=1)
            
            y_seg = data['GT']
            y_dtm = data['DTM']
            
            y_seg = y_seg.to(device)
            y_dtm = y_dtm.to(device)
            
            b, c, s1, s2, s3 = y_dtm.shape
            
            # Compute prediction and loss
            X = X.float()
            X = X.to(device)
            pred_seg, pred_dtm = model(X)
            pred_seg = sigmoid(pred_seg)
            
            print("Unique pred dtm")
            print(torch.max(softmax(pred_dtm)))
            print("Unique pred dtm")
            print(torch.unique(torch.argmax(softmax(pred_dtm), dim=1)))
            print("Unique y dtm")
            print(torch.unique(y_dtm.long()))

            loss_dice = loss_0(pred_seg, y_seg)
            loss_bce = loss_1(pred_seg, y_seg)
            y_dtm = torch.clamp(y_dtm.view(b, s1, s2, s3).long(), min=0, max=K - 1)
            loss_dtm = loss_ce_dtm(pred_dtm, y_dtm)
            #loss_dist = loss_term(pred_dtm, y_dtm)
            loss = 0.5 * (loss_dice + loss_bce) + 0.5 * loss_dtm #+ loss_dist
            train_loss += loss.item()
            train_loss_dice += loss_dice.item()
            train_loss_bce += loss_bce.item()
            train_loss_dtm += loss_dtm.item()

            pred_seg = nn.functional.threshold(pred_seg, threshold=0.5, value=0)
            ones = torch.ones(pred_seg.shape, dtype=torch.float)
            ones = ones.to(device)
            pred = torch.where(pred_seg > 0, ones, pred_seg)
            dice = dice_metric_pytorch(pred_seg, y_seg)
            dice = dice.cpu().detach().numpy()
            dice = np.mean(dice)
            
            # debug_dtm = torch.argmax(softmax(pred_dtm), dim=1)
            # print("Sum dtm")
            # print(torch.sum(debug_dtm))
            # print("Unique dtm")
            # print(torch.unique(debug_dtm))
            

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss = train_loss / len(dataloader)
        train_loss_dice = train_loss_dice / len(dataloader)
        train_loss_bce = train_loss_bce / len(dataloader)
        train_loss_dtm = train_loss_dtm / len(dataloader)
        end = time.time()
        epoch_duration = end - start
       
        logs = {"train_loss": train_loss,
                "loss_dice": train_loss_dice,
                "loss_bce": train_loss_bce,
                "loss_dtm": train_loss_dtm,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "epoch_duration": epoch_duration}
        
        print(f"loss: {train_loss:>7f}, val_loss:{val_loss:>7f}, dice_val:{val_dice:>7f}, dice:{dice:>7f}, time:{epoch_duration:>7f}")
        
        return logs
   
    
if __name__ == '__main__':
    dir_data = '/home/rouge/Documents/Thèse_Rougé_Pierre/Data/Bullit/raw/GT/*'
    
    for f in glob(dir_data):
        image = nib.load(f)
        data = image.get_fdata()
            
        dtm = distance_transform(data)
        
        dtm_nii = nib.Nifti1Image(dtm, affine=image.affine, header=image.header)
        
        nib.save(dtm_nii, f.replace('GT', 'DistanceMap'))
        
        print(np.unique(dtm))
        print(np.sum(dtm))
        # print("Number of radius=0")
        # print(np.sum(dtm==0))
        # print("Number of radius=1")
        # print(np.sum(dtm==1))
        # print("Number of radius=2")
        # print(np.sum(dtm==2))
        # print("Number of radius=3")
        # print(np.sum(dtm==3))
        # print("Number of radius=4")
        # print(np.sum(dtm==4))
        # print("Number of radius=5")
        # print(np.sum(dtm==5))
        # print("Number of radius=6")
        # print(np.sum(dtm==6))
        # print("Number of radius=7")
        # print(np.sum(dtm==7))
