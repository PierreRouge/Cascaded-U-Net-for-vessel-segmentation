#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:35:04 2021

@author: rouge
"""

import numpy as np
import torch
from torch import nn
import time
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils.utils_pytorch import dice_metric_pytorch, cl_dice_loss, dice_loss_pytorch, sigmoid_clip, cl_dice_loss_new_skeletonization, get_lr
from utils.utils_multitask import loss_dice_multitask, loss_CE_multitask, loss_dice_CE_multitask, loss_dice_cldice_multitask
from utils.utils_losses import msloss, fvloss

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

sigmoid = sigmoid_clip(1E-3)


# Training fonction for U-Net
def train_loop(dataloader, validloader, model, loss_param, patch_size, input_, optimizer, device, epoch, max_epoch, alpha_=0.5):

    if loss_param == "Dice":
        loss_0 = dice_loss_pytorch
        
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
                
                y_val = data['GT']
                
                X_val = X_val.float()
                X_val = X_val.to(device)
                
                pred_val = model(X_val)
                pred_val = sigmoid(pred_val)
    
                y_val = y_val.to(device)
                val_loss = loss_0(pred_val, y_val)
                val_loss = val_loss.item()
                val_loss_0 += val_loss
                pred_val = nn.functional.threshold(pred_val, threshold=0.5, value=0)
                ones = torch.ones(pred_val.shape, dtype=torch.float)
                ones = ones.to(device)
                pred_val = torch.where(pred_val > 0, ones, pred_val)
                dice_val = dice_metric_pytorch(pred_val, y_val)
                dice_val = dice_val.cpu().detach().numpy()
                dice_val = np.mean(dice_val)
                val_dice_0 += dice_val.item()
    
            val_loss = val_loss_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            
        train_loss = 0.0
        train_dice = 0.0
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
            
            y = data['GT']
            y = y.to(device)
            
            # Compute prediction and loss
            X = X.float()
            X = X.to(device)
            pred = model(X)
            pred = sigmoid(pred)
            loss = loss_0(pred, y)
            train_loss += loss.item()

            pred = nn.functional.threshold(pred, threshold=0.5, value=0)
            ones = torch.ones(pred.shape, dtype=torch.float)
            ones = ones.to(device)
            pred = torch.where(pred > 0, ones, pred)
            dice = dice_metric_pytorch(pred, y)
            dice = dice.cpu().detach().numpy()
            dice = np.mean(dice)
            train_dice += dice.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_norm = 0
            for p in model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(f"Total norm:{total_norm}")
            optimizer.step()
              
        train_loss = train_loss / len(dataloader)
        train_dice = train_dice / len(dataloader)
        end = time.time()
        epoch_duration = end - start
       
        logs = {"train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "epoch_duration": epoch_duration}
        
        print(f"loss: {train_loss:>7f}, val_loss:{val_loss:>7f}, dice_val:{val_dice:>7f}, dice_train:{train_dice:>7f}, time:{epoch_duration:>7f}")
        
        return logs
    
    if loss_param == "BCE":
        
        # pos_weight = torch.ones(patch_size, device=device) * 100
        # loss_0 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_0 = nn.BCEWithLogitsLoss()
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
                
                y_val = data['GT']
                
                X_val = X_val.float()
                X_val = X_val.to(device)
    
                pred_val = model(X_val)
    
                y_val = y_val.to(device)
                val_loss = loss_0(pred_val, y_val)
                val_loss = val_loss.item()
                val_loss_0 += val_loss
                
                pred_val = sigmoid(pred_val)
                pred_val = nn.functional.threshold(pred_val, threshold=0.5, value=0)
                ones = torch.ones(pred_val.shape, dtype=torch.float)
                ones = ones.to(device)
                pred_val = torch.where(pred_val > 0, ones, pred_val)
                dice_val = dice_metric_pytorch(pred_val, y_val)
                dice_val = dice_val.cpu().detach().numpy()
                dice_val = np.mean(dice_val)
                val_dice_0 += dice_val.item()
                
                # print(dice_val.item())
                # image = pred_val[0,0,:,:,32].detach().cpu().numpy()
                # plt.figure()
                # plt.imshow(image)
                # plt.show()
    
            val_loss = val_loss_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            
        train_loss = 0.0
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
                
            y = data['GT']
            
            # Compute prediction and loss
            X = X.float()
            X = X.to(device)
            pred = model(X)

            y = y.to(device)
            loss = loss_0(pred, y)
            train_loss += loss.item()

            pred = sigmoid(pred)
            pred = nn.functional.threshold(pred, threshold=0.5, value=0)
            ones = torch.ones(pred.shape, dtype=torch.float)
            ones = ones.to(device)
            pred = torch.where(pred > 0, ones, pred)
            dice = dice_metric_pytorch(pred, y)
            dice = dice.cpu().detach().numpy()
            dice = np.mean(dice)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(f"Total norm:{total_norm}")
            
            optimizer.step()
            
        train_loss = train_loss / len(dataloader)
        end = time.time()
        epoch_duration = end - start
        
        logs = {"train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "epoch_duration": epoch_duration}
        
        print(f"loss: {train_loss:>7f}, val_loss:{val_loss:>7f}, dice_val:{val_dice:>7f}, dice:{dice:>7f}, time:{epoch_duration:>7f}")
        
        return logs
        
    if loss_param == "Both":
        loss_1 = dice_loss_pytorch
        # pos_weight = torch.ones(patch_size, device=device) * 100
        # loss_2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_2 = nn.BCEWithLogitsLoss()
        
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
                
                y_val = data['GT']
                
                X_val = X_val.float()
                X_val = X_val.to(device)
                y_val = y_val.float()
                y_val = y_val.to(device)
                
                pred_val = model(X_val)
        
                val_loss = loss_1(sigmoid(pred_val), y_val) + loss_2(pred_val, y_val)
                val_loss = val_loss.item()
                val_loss_0 += val_loss
                
                pred_val = sigmoid(pred_val)
                pred_val = nn.functional.threshold(pred_val, threshold=0.5, value=0)
                ones = torch.ones(pred_val.shape, dtype=torch.float, device=device)
                pred_val = torch.where(pred_val > 0, ones, pred_val)
                dice_val = dice_metric_pytorch(pred_val, y_val)
                dice_val = torch.mean(dice_val)
                val_dice_0 += dice_val.item()
                
            val_loss = val_loss_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            
        loss_dice = 0.0
        loss_bce = 0.0
        train_loss = 0.0
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
            y = data['GT']
        
            # Compute prediction and loss
            X = X.float()
            X = X.to(device)
            y = y.float()
            y = y.to(device)
            
            pred = model(X)
           
            l_dice = loss_1(sigmoid(pred), y)
            l_bce = loss_2(pred, y)
            loss = 0.5 * l_dice + 0.5 * l_bce
            
            pred = sigmoid(pred)
            train_loss += loss.item()
            loss_dice += l_dice.item()
            loss_bce += l_bce.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(f"Total norm:{total_norm}")
            
            optimizer.step()
         
        train_loss = train_loss / len(dataloader)
        loss_dice = loss_dice / len(dataloader)
        loss_bce = loss_bce / len(dataloader)
        end = time.time()
        epoch_duration = end - start
        
        print(f"loss: {train_loss:>7f}, loss_dice: {loss_dice:>7f}, loss_bce: {loss_bce:>7f}, val_loss:{val_loss:>7f}, dice_val:{val_dice:>7f}, time:{epoch_duration:>7f}")
        # print(loss_dice)
        # print(loss_bce)
        logs = {"train_loss": train_loss,
                "loss_dice": loss_dice,
                "loss_bce": loss_bce,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "epoch_duration": epoch_duration
                }
        
        return logs
               
    if loss_param == 'clDice':
        print("clDice_on")
        if epoch < 100:
            alpha = 1 - (epoch/100) * (1 - alpha_)
        else:
            alpha = alpha_
        loss_0 = cl_dice_loss(alpha , iter_=3)
        model.eval()
        val_loss_0 = 0.0
        val_dice_0 = 0.0
        val_cldice_loss_0 = 0.0
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
                
                y_val = data['GT']
                
                X_val = X_val.float()
                X_val = X_val.to(device)
                y_val = y_val.float()
                y_val = y_val.to(device)
            
    
                pred_val = model(X_val)
                pred_val = sigmoid(pred_val)
               
                val_loss, val_dice_loss, val_cldice_loss = loss_0(pred_val, y_val)
                val_loss = val_loss.item()
                val_loss_0 += val_loss
                pred_val = nn.functional.threshold(pred_val, threshold=0.5, value=0)
                ones = torch.ones(pred_val.shape, dtype=torch.float)
                ones = ones.to(device)
                pred_val = torch.where(pred_val > 0, ones, pred_val)
                dice_val = dice_metric_pytorch(pred_val, y_val)
                dice_val = dice_val.cpu().detach().numpy()
                dice_val = np.mean(dice_val)
                val_dice_0 += dice_val
                
                val_cldice_loss_0 += val_cldice_loss.item()
    
            val_loss = val_loss_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            val_cldice = val_cldice_loss_0 / len(validloader)
            
        train_loss = 0.0
        train_dice_loss= 0.0
        train_cldice_loss = 0.0
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
            y = data['GT']
            
            # Compute prediction and loss
            X = X.float()
            X = X.to(device)
            y = y.float()
            y = y.to(device)
            
            pred = model(X)
            pred = sigmoid(pred)

            y = y.to(device)
            loss, dice_loss, cl_dice_loss_ = loss_0(pred, y)
            train_loss += loss.item()
            train_dice_loss += dice_loss.item()
            train_cldice_loss += cl_dice_loss_.item()
            

            pred = nn.functional.threshold(pred, threshold=0.5, value=0)
            ones = torch.ones(pred.shape, dtype=torch.float)
            ones = ones.to(device)
            pred = torch.where(pred > 0, ones, pred)
            dice = dice_metric_pytorch(pred, y)
            dice = dice.cpu().detach().numpy()
            dice = np.mean(dice)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss = train_loss / len(dataloader) 
        train_dice_loss = train_dice_loss / len(dataloader) 
        train_cldice_loss = train_cldice_loss / len(dataloader) 
        end = time.time()
        epoch_duration = end - start 
        
        logs = {"train_loss": train_loss,
                "train_dice_loss": train_dice_loss,
                "train_cldice_loss": train_cldice_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_cldice_loss": val_cldice,
                "epoch_duration": epoch_duration
            }
        
        print(f"loss: {train_loss:>7f}, dice_loss: {train_dice_loss:>7f}, train_cldice_loss: {train_loss:>7f}, val_loss:{val_loss:>7f}, dice_val:{val_dice:>7f}, dice:{dice:>7f}, time:{epoch_duration:>7f}")
         
        return logs
    
    if loss_param == 'tsloss':
        loss_1 = dice_loss_pytorch
        loss_2 = nn.BCELoss()
        
        # weigth morphological loss
        wms = 0.05
        
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
                
                y_val = data['GT']
                
                X_val = X_val.float()
                X_val = X_val.to(device)
                y_val = y_val.float()
                y_val = y_val.to(device)
                
                pred_val = model(X_val)
                pred_val = sigmoid(pred_val)
    
                val_loss = loss_1(pred_val, y_val) + loss_2(pred_val, y_val)
                val_loss = val_loss.item()
                val_loss_0 += val_loss
                pred_val = nn.functional.threshold(pred_val, threshold=0.5, value=0)
                ones = torch.ones(pred_val.shape, dtype=torch.float, device=device)
                pred_val = torch.where(pred_val > 0, ones, pred_val)
                dice_val = dice_metric_pytorch(pred_val, y_val)
                dice_val = torch.mean(dice_val)
                val_dice_0 += dice_val.item()
                
            val_loss = val_loss_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            
        loss_Dice = 0.0
        loss_BCE = 0.0
        train_loss = 0.0
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
            y = data['GT']
        
            # Compute prediction and loss
            X = X.float()
            X = X.to(device)
            y = y.float()
            y = y.to(device)
            
            pred = model(X)
            pred = sigmoid(pred)

           
            loss_dice = loss_1(pred, y) 
            loss_bce = loss_2(pred, y)
            loss_frangi = fvloss([pred], y)
            loss_morpho = msloss([pred], y)
            
            loss = loss_dice + loss_bce + loss_frangi + wms * loss_morpho
            
            train_loss += loss_dice.item() + loss_bce.item() + loss_frangi.item() + wms * loss_morpho.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         
        train_loss = train_loss / len(dataloader)
        end = time.time()
        epoch_duration = end - start 
        
        print(f"loss: {train_loss:>7f}, val_loss:{val_loss:>7f}, dice_val:{val_dice:>7f}, time:{epoch_duration:>7f}")
        
        logs = {"train_loss": train_loss,
                "loss_dice": loss_dice.item(),
                "loss_bce": loss_bce.item(),
                "loss_frangi": loss_frangi.item(),
                "loss_morpho":  loss_morpho.item(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "epoch_duration": epoch_duration
            }
        
        return logs
