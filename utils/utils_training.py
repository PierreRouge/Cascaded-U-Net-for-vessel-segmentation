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
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils.utils_pytorch import dice_metric_pytorch, cl_dice_loss, dice_loss_pytorch, sigmoid_clip, cl_dice_loss_new_skeletonization, get_lr
from utils.utils_multitask import loss_dice_multitask, loss_CE_multitask, loss_dice_CE_multitask, loss_dice_cldice_multitask

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

sigmoid = sigmoid_clip(1E-3)


# Training fonction for U-Net
def train_loop(dataloader, validloader, model, loss_param, input_, optimizer, device, epoch, max_epoch, alpha_=0.5):

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
            pred = sigmoid(pred)

            y = y.to(device)
            loss = loss_0(pred, y)
            train_loss += loss.item()

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
        end = time.time()
        epoch_duration = end - start
       
        logs = {"train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "epoch_duration": epoch_duration}
        
        print(f"loss: {train_loss:>7f}, val_loss:{val_loss:>7f}, dice_val:{val_dice:>7f}, dice:{dice:>7f}, time:{epoch_duration:>7f}")
        
        return logs
    
    if loss_param == "BCE":
        loss_0 = nn.BCELoss()
        
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
            pred = sigmoid(pred)

            y = y.to(device)
            loss = loss_0(pred, y)
            train_loss += loss.item()

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
        loss_2 = nn.BCELoss()
        
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

           
            loss = loss_1(pred, y) + loss_2(pred, y)
            train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         
        train_loss = train_loss / len(dataloader)
        end = time.time()
        epoch_duration = end - start 
        
        print(f"loss: {train_loss:>7f}, val_loss:{val_loss:>7f}, dice_val:{val_dice:>7f}, time:{epoch_duration:>7f}")
        
        logs = {"train_loss": train_loss,
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
   
    
# Training function for U-Net with Boolean and Euler skeletonization
def train_loop_newskel(dataloader, validloader, model, input_, optimizer, device, epoch, max_epoch, alpha_=0.5, beta=0.33, tau=1.0, method='Boolean', num_iter=5):

    loss_0 = cl_dice_loss_new_skeletonization(alpha=alpha_, beta=beta, tau=tau, method=method, num_iter=num_iter)
    
    model.eval()
    val_loss_0 = 0.0
    val_dice_0 = 0.0
    val_loss_dice_mean = 0.0
    val_loss_cldice_mean = 0.0
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
            
            val_loss, val_loss_dice, val_loss_cldice = loss_0(pred_val, y_val)
            
            val_loss = val_loss.item()
            val_loss_0 += val_loss
            
            val_loss_dice_mean += val_loss_dice.item()
            val_loss_cldice_mean += val_loss_cldice.item()
            
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
        val_loss_dice_mean = val_loss_dice_mean / len(validloader)
        val_loss_cldice_mean = val_loss_cldice_mean / len(validloader)
        
    train_loss = 0.0
    train_loss_dice_mean = 0.0
    train_loss_cldice_mean = 0.0
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
        pred = sigmoid(pred)

        y = y.to(device)
        
        loss, loss_dice, loss_cldice = loss_0(pred, y)
        
        train_loss += loss.item()
        train_loss_dice_mean += loss_dice.item()
        train_loss_cldice_mean += loss_cldice.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss = train_loss / len(dataloader)
    train_loss_dice_mean = train_loss_dice_mean / len(dataloader)
    train_loss_cldice_mean = train_loss_cldice_mean / len(dataloader)
    
    end = time.time()
    epoch_duration = end - start
   
    logs = {"train_loss": train_loss,
            "train_loss_dice": train_loss_dice_mean,
            "train_loss_cldice": train_loss_cldice_mean,
            "val_loss": val_loss,
            "val_loss_dice": val_loss_dice_mean,
            "val_loss_cldice": val_loss_cldice_mean,
            "val_dice": val_dice,
            "epoch_duration": epoch_duration}
    
    print(f"loss: {train_loss:>7f}, val_loss:{val_loss:>7f}, dice_val:{val_dice:>7f}, time:{epoch_duration:>7f}")
    
    return logs


# Training functions for cascaded U-Net
def train_loop_cascaded(dataloader, validloader, model, optimizer, loss_param, alpha, device, alpha_dice_ce=0.5, lambda1=0.2, lambda2=0.5, lambda3=0.1):
    
    sigmoid = sigmoid_clip(1e-5)
    
    if loss_param == "Dice":
        loss_0 = loss_dice_multitask(alpha)
        
        model.eval()
        val_loss_0 = 0.0
        val_loss_seg_0 = 0.0
        val_loss_skel_0 = 0.0
        val_dice_0 = 0.0
        with torch.no_grad():
            for batch, data in enumerate(validloader):
                X_val = data['image']
                y_val = data['GT']
                y_val_skel = data['skeleton']
                X_val = X_val.float()
                X_val = X_val.to(device)

                pred_val, pred_val_skel = model(X_val)
                pred_val = sigmoid(pred_val)
                pred_val_skel = sigmoid(pred_val_skel)

                y_val = y_val.to(device)
                y_val_skel = y_val_skel.to(device)
                
                val_loss, val_loss_seg, val_loss_skel = loss_0(pred_val, pred_val_skel, y_val, y_val_skel)
                val_loss = val_loss.item()
                val_loss_seg = val_loss_seg.item()
                val_loss_skel = val_loss_skel.item()
                val_loss_0 += val_loss
                val_loss_seg_0 += val_loss_seg
                val_loss_skel_0 += val_loss_skel
                pred_val = nn.functional.threshold(pred_val, threshold=0.5, value=0)
                ones = torch.ones(pred_val.shape, dtype=torch.float, device=device)
                pred_val = torch.where(pred_val > 0, ones, pred_val)
                
                pred_val_skel = nn.functional.threshold(pred_val_skel, threshold=0.5, value=0)
                pred_val_skel = torch.where(pred_val_skel > 0, ones, pred_val)
                
                val_dice = dice_metric_pytorch(pred_val, y_val)
                val_dice = torch.mean(val_dice)
                val_dice_0 += val_dice

            val_loss = val_loss_0 / len(validloader)
            val_loss_seg = val_loss_seg_0 / len(validloader)
            val_loss_skel = val_loss_skel_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            
        train_loss = 0.0
        train_loss_seg = 0.0
        train_loss_skel = 0.0
        start = time.time()
        for batch_train, data in enumerate(dataloader):
            
            X = data['image']
            y = data['GT']
            y_skel = data['skeleton']
            X = X.float()
            X = X.to(device)
            pred, pred_skel = model(X)
            pred = sigmoid(pred)
            pred_skel = sigmoid(pred_skel)

            y = y.to(device)
            y_skel = y_skel.to(device)
            loss, loss_seg, loss_skel = loss_0(pred, pred_skel, y, y_skel)
            train_loss += loss.item()
            train_loss_seg += loss_seg.item()
            train_loss_skel += loss_skel.item()

            pred = nn.functional.threshold(pred, threshold=0.5, value=0)
            ones = torch.ones(pred.shape, dtype=torch.float, device=device)
            pred = torch.where(pred > 0, ones, pred)
            dice = dice_metric_pytorch(pred, y)
            dice = torch.mean(dice)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss = train_loss / len(dataloader)
        train_loss_seg = train_loss_seg / len(dataloader)
        train_loss_skel = train_loss_skel / len(dataloader)
        end = time.time()
        epoch_duration = end - start
       
        print(f"loss: {train_loss:>7f}, loss_seg: {train_loss_seg:>7f}, loss_skel: {train_loss_skel:>7f}, val_loss:{val_loss:>7f}, val_dice:{val_dice:>7f}, time:{epoch_duration:>7f}")
            
        return train_loss, train_loss_seg, train_loss_skel, val_loss, val_dice, epoch_duration, X_val, y_val, y_val_skel, pred_val, pred_val_skel
        
    if loss_param == 'CE':
        loss_0 = loss_CE_multitask(alpha)
            
        model.eval()
        val_loss_0 = 0.0
        val_loss_seg_0 = 0.0
        val_loss_skel_0 = 0.0
        val_dice_0 = 0.0
        with torch.no_grad():
            for batch, data in enumerate(validloader):
                X_val = data['image']
                y_val = data['GT']
                y_val_skel = data['skeleton']
                X_val = X_val.float()
                X_val = X_val.to(device)

                pred_val, pred_val_skel = model(X_val)
                pred_val = sigmoid(pred_val)
                pred_val_skel = sigmoid(pred_val_skel)

                y_val = y_val.to(device)
                y_val_skel = y_val_skel.to(device)
                
                val_loss, val_loss_seg, val_loss_skel = loss_0(pred_val, pred_val_skel, y_val, y_val_skel)
                val_loss = val_loss.item()
                val_loss_seg = val_loss_seg.item()
                val_loss_skel = val_loss_skel.item()
                val_loss_0 += val_loss
                val_loss_seg_0 += val_loss_seg
                val_loss_skel_0 += val_loss_skel
                pred_val = nn.functional.threshold(pred_val, threshold=0.5, value=0)
                ones = torch.ones(pred_val.shape, dtype=torch.float, device=device)
                pred_val = torch.where(pred_val > 0, ones, pred_val)
                
                pred_val_skel = nn.functional.threshold(pred_val_skel, threshold=0.5, value=0)
                pred_val_skel = torch.where(pred_val_skel > 0, ones, pred_val)
                
                val_dice = dice_metric_pytorch(pred_val, y_val)
                val_dice = torch.mean(val_dice)
                val_dice_0 += val_dice

            val_loss = val_loss_0 / len(validloader)
            val_loss_seg = val_loss_seg_0 / len(validloader)
            val_loss_skel = val_loss_skel_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            
        train_loss = 0.0
        train_loss_seg = 0.0
        train_loss_skel = 0.0
        start = time.time()
        for batch_train, data in enumerate(dataloader):
            X = data['image']
            y = data['GT']
            y_skel = data['skeleton']
            X = X.float()
            X = X.to(device)
            pred, pred_skel = model(X)
            pred = sigmoid(pred)
            pred_skel = sigmoid(pred_skel)

            y = y.to(device)
            y_skel = y_skel.to(device)
            loss, loss_seg, loss_skel = loss_0(pred, pred_skel, y, y_skel)
            train_loss += loss.item()
            train_loss_seg += loss_seg.item()
            train_loss_skel += loss_skel.item()

            pred = nn.functional.threshold(pred, threshold=0.5, value=0)
            ones = torch.ones(pred.shape, dtype=torch.float, device=device)
            pred = torch.where(pred > 0, ones, pred)
            dice = dice_metric_pytorch(pred, y)
            dice = torch.mean(dice)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss = train_loss / len(dataloader)
        train_loss_seg = train_loss_seg / len(dataloader)
        train_loss_skel = train_loss_skel / len(dataloader)
        end = time.time()
        epoch_duration = end - start
       
        print(f"loss: {train_loss:>7f}, loss_seg: {train_loss_seg:>7f}, loss_skel: {train_loss_skel:>7f}, val_loss:{val_loss:>7f}, val_dice:{val_dice:>7f}, time:{epoch_duration:>7f}")
            
        return train_loss, train_loss_seg, train_loss_skel, val_loss, val_dice, epoch_duration, X_val, y_val, y_val_skel, pred_val, pred_val_skel
    
    if loss_param == "Both":
        loss_0 = loss_dice_CE_multitask(alpha, alpha_dice_ce)
            
        model.eval()
        val_loss_0 = 0.0
        val_loss_seg_0 = 0.0
        val_loss_skel_0 = 0.0
        val_dice_0 = 0.0
        with torch.no_grad():
            for batch, data in enumerate(validloader):
                X_val = data['image']
                y_val = data['GT']
                y_val_skel = data['skeleton']
                X_val = X_val.float()
                X_val = X_val.to(device)

                pred_val, pred_val_skel = model(X_val)
                pred_val = sigmoid(pred_val)
                pred_val_skel = sigmoid(pred_val_skel)

                y_val = y_val.to(device)
                y_val_skel = y_val_skel.to(device)
                
                val_loss, val_loss_seg, val_loss_skel, val_dice_loss, val_CE_loss, val_dice_loss_skel, val_CE_loss_skel = loss_0(pred_val, pred_val_skel, y_val, y_val_skel)
                val_loss = val_loss.item()
                val_loss_seg = val_loss_seg.item()
                val_loss_skel = val_loss_skel.item()
                val_loss_0 += val_loss
                val_loss_seg_0 += val_loss_seg
                val_loss_skel_0 += val_loss_skel
                pred_val = nn.functional.threshold(pred_val, threshold=0.5, value=0)
                ones = torch.ones(pred_val.shape, dtype=torch.float, device=device)
                pred_val = torch.where(pred_val > 0, ones, pred_val)
                
                pred_val_skel = nn.functional.threshold(pred_val_skel, threshold=0.5, value=0)
                pred_val_skel = torch.where(pred_val_skel > 0, ones, pred_val)
                
                val_dice = dice_metric_pytorch(pred_val, y_val)
                val_dice = torch.mean(val_dice)
                val_dice_0 += val_dice

            val_loss = val_loss_0 / len(validloader)
            val_loss_seg = val_loss_seg_0 / len(validloader)
            val_loss_skel = val_loss_skel_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            
        train_loss = 0.0
        train_loss_seg = 0.0
        train_loss_skel = 0.0
        train_dice_loss = 0.0
        train_CE_loss = 0.0
        train_dice_loss_skel = 0.0
        train_CE_loss_skel = 0.0
        start = time.time()
        for batch_train, data in enumerate(dataloader):
            X = data['image']
            y = data['GT']
            y_skel = data['skeleton']
            X = X.float()
            X = X.to(device)
            pred, pred_skel = model(X)
            pred = sigmoid(pred)
            pred_skel = sigmoid(pred_skel)

            y = y.to(device)
            y_skel = y_skel.to(device)
            loss, loss_seg, loss_skel, dice_loss, CE_loss, dice_loss_skel, CE_loss_skel = loss_0(pred, pred_skel, y, y_skel)
            train_loss += loss.item()
            train_loss_seg += loss_seg.item()
            train_loss_skel += loss_skel.item()
            train_dice_loss += dice_loss.item()
            train_CE_loss += CE_loss.item()
            train_dice_loss_skel += dice_loss_skel.item()
            train_CE_loss_skel += CE_loss_skel.item()
            
            pred = nn.functional.threshold(pred, threshold=0.5, value=0)
            ones = torch.ones(pred.shape, dtype=torch.float, device=device)
            pred = torch.where(pred > 0, ones, pred)
            dice = dice_metric_pytorch(pred, y)
            dice = torch.mean(dice)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss = train_loss / len(dataloader)
        train_loss_seg = train_loss_seg / len(dataloader)
        train_loss_skel = train_loss_skel / len(dataloader)
        train_dice_loss = train_dice_loss / len(dataloader)
        train_CE_loss = train_CE_loss / len(dataloader)
        train_dice_loss_skel = train_dice_loss_skel / len(dataloader)
        train_CE_loss_skel = train_CE_loss_skel / len(dataloader)
        end = time.time()
        epoch_duration = end - start
       
        print(f"loss: {train_loss:>7f}, loss_seg: {train_loss_seg:>7f}, loss_skel: {train_loss_skel:>7f}, val_loss:{val_loss:>7f}, val_dice:{val_dice:>7f}, time:{epoch_duration:>7f}")
            
        return train_loss, train_loss_seg, train_loss_skel, train_dice_loss, train_CE_loss, train_dice_loss_skel, train_CE_loss_skel, val_loss, val_dice, epoch_duration, X_val, y_val, y_val_skel, pred_val, pred_val_skel

    if loss_param == "clDice":
        loss_0 = loss_dice_cldice_multitask(lambda1, lambda2, lambda3)
        
        model.eval()
        val_loss_0 = 0.0
        val_loss_seg_0 = 0.0
        val_loss_skel_0 = 0.0
        val_loss_cldice_0 = 0.0
        val_dice_0 = 0.0
        with torch.no_grad():
            for batch, data in enumerate(validloader):
                X_val = data['image']
                y_val = data['GT']
                y_val_skel = data['skeleton']
                X_val = X_val.float()
                X_val = X_val.to(device)

                pred_val, pred_val_skel = model(X_val)
                pred_val = sigmoid(pred_val)
                pred_val_skel = sigmoid(pred_val_skel)

                y_val = y_val.to(device)
                y_val_skel = y_val_skel.to(device)
                
                val_loss, val_loss_seg, val_loss_skel, val_loss_cldice = loss_0(pred_val, pred_val_skel, y_val, y_val_skel)
                val_loss = val_loss.item()
                val_loss_seg = val_loss_seg.item()
                val_loss_skel = val_loss_skel.item()
                val_loss_cldice = val_loss_cldice.item()
                val_loss_0 += val_loss
                val_loss_seg_0 += val_loss_seg
                val_loss_skel_0 += val_loss_skel
                val_loss_cldice_0 += val_loss_cldice
                pred_val = nn.functional.threshold(pred_val, threshold=0.5, value=0)
                ones = torch.ones(pred_val.shape, dtype=torch.float, device=device)
                pred_val = torch.where(pred_val > 0, ones, pred_val)
                
                pred_val_skel = nn.functional.threshold(pred_val_skel, threshold=0.5, value=0)
                pred_val_skel = torch.where(pred_val_skel > 0, ones, pred_val)
                
                val_dice = dice_metric_pytorch(pred_val, y_val)
                val_dice = torch.mean(val_dice)
                val_dice_0 += val_dice.item()

            val_loss = val_loss_0 / len(validloader)
            val_loss_seg = val_loss_seg_0 / len(validloader)
            val_loss_skel = val_loss_skel_0 / len(validloader)
            val_loss_cldice = val_loss_cldice_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            
        train_loss = 0.0
        train_loss_seg = 0.0
        train_loss_skel = 0.0
        train_loss_cldice = 0.0
        start = time.time()
        for batch_train, data in enumerate(dataloader):
            X = data['image']
            y = data['GT']
            y_skel = data['skeleton']
            X = X.float()
            X = X.to(device)
            pred, pred_skel = model(X)
            pred = sigmoid(pred)
            pred_skel = sigmoid(pred_skel)

            y = y.to(device)
            y_skel = y_skel.to(device)
            loss, loss_seg, loss_skel, loss_cldice = loss_0(pred, pred_skel, y, y_skel)
            train_loss += loss.item()
            train_loss_seg += loss_seg.item()
            train_loss_skel += loss_skel.item()
            train_loss_cldice += loss_cldice.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss = train_loss / len(dataloader)
        train_loss_seg = train_loss_seg / len(dataloader)
        train_loss_skel = train_loss_skel / len(dataloader)
        train_loss_cldice = train_loss_cldice / len(dataloader)
        end = time.time()
        epoch_duration = end - start
       
        logs = {"train_loss": train_loss,
                "train_loss_seg": train_loss_seg,
                "train_loss_skel": train_loss_skel,
                "train_loss_cldice": train_loss_cldice,
                "val_loss": val_loss,
                "val_loss_seg": val_loss_seg,
                "val_loss_skel": val_loss_skel,
                "val_loss_cldice": val_loss_cldice,
                "val_dice": val_dice,
                "epoch_duration": epoch_duration}
        
        print(f"loss: {train_loss:>7f}, loss_seg: {train_loss_seg:>7f}, loss_skel: {train_loss_skel:>7f}, val_loss:{val_loss:>7f}, val_dice:{val_dice:>7f}, time:{epoch_duration:>7f}")
            
        return logs

    
def training_cascaded(dataset_train, dataset_val, model, optimizer, epochs, batch_size, device, res, lambda1=0.2, lambda2=0.5, lambda3=0.1, loss_param='custom', alpha=0.8, scheduler=None, alpha_dice_ce=0.5, loss_scheduling=False, train_size_epoch=250, val_size_epoch=50):
    
    if loss_param == 'Dice' or loss_param == 'CE':
        # Training phase
        train_history = []
        val_history = []
        val_dice_history = []
        loss_seg_history = []
        loss_skel_history = []
        lr_history = []
        
        print(f"Epoch {1}\n-------------------------------")
        
        indices_train = np.random.choice(np.arange(0, len(dataset_train)), size=train_size_epoch)
        sampler_train = SubsetRandomSampler(indices_train)
        indices_val = np.random.choice(np.arange(0, len(dataset_val)), size=val_size_epoch)
        sampler_val = SubsetRandomSampler(indices_val)
        
        train_data = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers=4, pin_memory=True)
        val_data = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, num_workers=4, pin_memory=True)
        
        epoch_save = 1
        train_loss, train_loss_seg, train_loss_skel, val_loss, val_dice, epoch_duration, X_val, y_val, y_val_skel, pred_val, pred_val_skel = train_loop_cascaded(train_data, val_data, model, optimizer, loss_param, alpha, device, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
        val_loss_save = val_loss
        train_history.append(train_loss)
        val_history.append(val_loss)
        val_dice_history.append(val_dice)
        loss_seg_history.append(train_loss_seg)
        loss_skel_history.append(train_loss_skel)
        
        file_training = open(res + "/training.txt", "a")
        file_training.write("loss:" + str(train_loss) + ',loss_seg:' + str(train_loss_seg) + ',loss_skel:' + str(train_loss_skel) + ',val_loss:' + str(val_loss) + ',val_dice:' + str(val_dice) + ',time:' + str(epoch_duration) + ',alpha_dice:' + str(alpha_dice_ce) + '\n')
        file_training.close()
        
        if scheduler is not None:
            lr = scheduler.get_last_lr()
            lr_history.append(lr)
            scheduler.step()
            
        else:
            lr = [get_lr(optimizer)]
            lr_history.append(lr)
        
        log_dictionnary = {
            "Loss": train_loss,
            "Loss segmentation": train_loss_seg,
            "Loss centerline": train_loss_skel,
            "Validation Loss": val_loss,
            "Validation Dice metric": val_dice,
            "Learning Rate": float(lr[0]),
            "Epoch": 1}
        wandb.log(log_dictionnary)
        
        for t in range(1, epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            indices_train = np.random.choice(np.arange(0, len(dataset_train)), size=train_size_epoch)
            sampler_train = SubsetRandomSampler(indices_train)
            indices_val = np.random.choice(np.arange(0, len(dataset_val)), size=val_size_epoch)
            sampler_val = SubsetRandomSampler(indices_val)
            
            train_data = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers=4, pin_memory=True)
            val_data = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, num_workers=4, pin_memory=True)
            
            if loss_scheduling:
                if t < int(epochs / 2):
                    alpha_dice_ce = (t / epochs) + 0.5
                else:
                    alpha_dice_ce = 1.
        
            train_loss, train_loss_seg, train_loss_skel, val_loss, val_dice, epoch_duration, X_val, y_val, y_val_skel, pred_val, pred_val_skel = train_loop_cascaded(train_data, val_data, model, optimizer, loss_param, alpha, device, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
            train_history.append(train_loss)
            val_history.append(val_loss)
            val_dice_history.append(val_dice)
            loss_seg_history.append(train_loss_seg)
            loss_skel_history.append(train_loss_skel)
        
            file_training = open(res + "/training.txt", "a")
            file_training.write("loss:" + str(train_loss) + ',loss_seg:' + str(train_loss_seg) + ',loss_skel:' + str(train_loss_skel) + ',val_loss:' + str(val_loss) + ',val_dice:' + str(val_dice) + ',time:' + str(epoch_duration) + ',alpha_dice:' + str(alpha_dice_ce) + '\n')
            file_training.close()
            
            if scheduler is not None:
                lr = scheduler.get_last_lr()
                lr_history.append(lr)
                scheduler.step()
                
            else:
                lr = [get_lr(optimizer)]
                lr_history.append(lr)
            
            image_mra = wandb.data_types.Image(X_val[0, 0, :, :, 30], caption='MRA Image')
            gt = wandb.data_types.Image(y_val[0, 0, :, :, 30], caption='Segmentation ground truth')
            pred = wandb.data_types.Image(pred_val[0, 0, :, :, 30], caption='Segmentation prediction')
            centerline_gt = wandb.data_types.Image(y_val_skel[0, 0, :, :, 30], caption='Centerlines ground truth')
            centerline_pred = wandb.data_types.Image(pred_val_skel[0, 0, :, :, 30], caption='Centerlines prediction')
            maxi = torch.max(X_val[0, 0, :, :, 30])
            
            log_dictionnary = {
                "Loss": train_loss,
                "Loss segmentation": train_loss_seg,
                "Loss centerline": train_loss_skel,
                "Validation Loss": val_loss,
                "Validation Dice metric": val_dice,
                "Learning Rate": float(lr[0]),
                "Epoch": t + 1,
                "MRA Image": image_mra,
                "Segmentation gorund truth": gt,
                "Segmentation prediction": pred,
                "Centerlines ground truth": centerline_gt,
                "Centerlines prediction": centerline_pred,
                "Max intensity slice": maxi}

            if (val_loss < val_loss_save):
                torch.save(model, res + '/model.pth')
                torch.save(model.state_dict(), res + '/model_weights.pth')
                val_loss_save = val_loss
                epoch_save = t + 1
                
        print("Done!")
        return train_history, val_history, val_dice_history, loss_seg_history, loss_skel_history, lr_history, val_loss_save, epoch_save
    
    if loss_param == 'Both':
        # Training phase
        train_history = []
        val_history = []
        val_dice_history = []
        loss_seg_history = []
        loss_skel_history = []
        lr_history = []
        
        print(f"Epoch {1}\n-------------------------------")
        
        indices_train = np.random.choice(np.arange(0, len(dataset_train)), size=train_size_epoch)
        sampler_train = SubsetRandomSampler(indices_train)
        indices_val = np.random.choice(np.arange(0, len(dataset_val)), size=val_size_epoch)
        sampler_val = SubsetRandomSampler(indices_val)
        
        train_data = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers=4, pin_memory=True)
        val_data = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, num_workers=4, pin_memory=True)
        
        epoch_save = 1
        train_loss, train_loss_seg, train_loss_skel, train_dice_loss, train_CE_loss, train_dice_loss_skel, train_CE_loss_skel, val_loss, val_dice, epoch_duration, X_val, y_val, y_val_skel, pred_val, pred_val_skel = train_loop_cascaded(train_data, val_data, model, optimizer, loss_param, alpha, device, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
        val_loss_save = val_loss
        train_history.append(train_loss)
        val_history.append(val_loss)
        val_dice_history.append(val_dice)
        loss_seg_history.append(train_loss_seg)
        loss_skel_history.append(train_loss_skel)
        
        file_training = open(res + "/training.txt", "a")
        file_training.write("loss:" + str(train_loss) + ',loss_seg:' + str(train_loss_seg) + ',loss_skel:' + str(train_loss_skel) + ',val_loss:' + str(val_loss) + ',val_dice:' + str(val_dice) + ',time:' + str(epoch_duration) + ',alpha_dice:' + str(alpha_dice_ce) + '\n')
        file_training.close()
        
        if scheduler is not None:
            lr = scheduler.get_last_lr()
            lr_history.append(lr)
            scheduler.step()
            
        else:
            lr = [get_lr(optimizer)]
            lr_history.append(lr)
        
        log_dictionnary = {
            "Loss": train_loss,
            "Loss segmentation": train_loss_seg,
            "Loss centerline": train_loss_skel,
            "Validation Loss": val_loss,
            "Validation Dice metric": val_dice,
            "Train Dice Loss": train_dice_loss,
            "Train CE Loss": train_CE_loss,
            "Train Dice Loss Skeleton": train_dice_loss_skel,
            "Train CE Loss Skeleton": train_CE_loss_skel,
            "Learning Rate": float(lr[0]),
            "Epoch": 1}
        wandb.log(log_dictionnary)
        
        for t in range(1, epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            indices_train = np.random.choice(np.arange(0, len(dataset_train)), size=train_size_epoch)
            sampler_train = SubsetRandomSampler(indices_train)
            indices_val = np.random.choice(np.arange(0, len(dataset_val)), size=val_size_epoch)
            sampler_val = SubsetRandomSampler(indices_val)
            
            train_data = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers=4, pin_memory=True)
            val_data = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, num_workers=4, pin_memory=True)
            
            if loss_scheduling:
                if t < int(epochs / 2):
                    alpha_dice_ce = (t / epochs) + 0.5
                else:
                    alpha_dice_ce = 1.
        
            train_loss, train_loss_seg, train_loss_skel, train_dice_loss, train_CE_loss, train_dice_loss_skel, train_CE_loss_skel, val_loss, val_dice, epoch_duration, X_val, y_val, y_val_skel, pred_val, pred_val_skel = train_loop_cascaded(train_data, val_data, model, optimizer, loss_param, alpha, device, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
            train_history.append(train_loss)
            val_history.append(val_loss)
            val_dice_history.append(val_dice)
            loss_seg_history.append(train_loss_seg)
            loss_skel_history.append(train_loss_skel)
        
            file_training = open(res + "/training.txt", "a")
            file_training.write("loss:" + str(train_loss) + ',loss_seg:' + str(train_loss_seg) + ',loss_skel:' + str(train_loss_skel) + ',val_loss:' + str(val_loss) + ',val_dice:' + str(val_dice) + ',time:' + str(epoch_duration) + ',alpha_dice:' + str(alpha_dice_ce) + '\n')
            file_training.close()
        
            if scheduler is not None:
                lr = scheduler.get_last_lr()
                lr_history.append(lr)
                scheduler.step()
                
            else:
                lr = [get_lr(optimizer)]
                lr_history.append(lr)
            
            if (val_loss < val_loss_save):
                torch.save(model, res + '/model.pth')
                torch.save(model.state_dict(), res + '/model_weights.pth')
                val_loss_save = val_loss
                epoch_save = t + 1
                
            log_dictionnary = {
                "Loss": train_loss,
                "Loss segmentation": train_loss_seg,
                "Loss centerline": train_loss_skel,
                "Validation Loss": val_loss,
                "Validation Dice metric": val_dice,
                "Train Dice Loss": train_dice_loss,
                "Train CE Loss": train_CE_loss,
                "Train Dice Loss Skeleton": train_dice_loss_skel,
                "Train CE Loss Skeleton": train_CE_loss_skel,
                "Learning Rate": float(lr[0]),
                "Epoch": t + 1}
            wandb.log(log_dictionnary)
            
        print("Done!")
        return train_history, val_history, val_dice_history, loss_seg_history, loss_skel_history, lr_history, val_loss_save, epoch_save
    
    if loss_param == 'clDice':
        # Training phase
        train_history = []
        val_history = []
        val_dice_history = []
        loss_seg_history = []
        loss_skel_history = []
        loss_cldice_history = []
        lr_history = []
        
        print(f"Epoch {1}\n-------------------------------")
        
        indices_train = np.random.choice(np.arange(0, len(dataset_train)), size=train_size_epoch)
        sampler_train = SubsetRandomSampler(indices_train)
        indices_val = np.random.choice(np.arange(0, len(dataset_val)), size=val_size_epoch)
        sampler_val = SubsetRandomSampler(indices_val)
        
        train_data = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers=4, pin_memory=True)
        val_data = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, num_workers=4, pin_memory=True)
        
        epoch_save = 1
        logs = train_loop_cascaded(train_data, val_data, model, optimizer, loss_param, alpha, device, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
        
        train_loss = logs["train_loss"]
        train_loss_seg = logs["train_loss_seg"]
        train_loss_skel = logs["train_loss_skel"]
        train_loss_cldice = logs["train_loss_cldice"]
        val_loss = logs["val_loss"]
        val_loss_seg = logs["val_loss_seg"]
        val_loss_skel = logs["val_loss_skel"]
        val_loss_cldice = logs["val_loss_cldice"]
        val_dice = logs["val_dice"]
        epoch_duration = logs["epoch_duration"]
        
        val_loss_save = val_loss
        train_history.append(train_loss)
        val_history.append(val_loss)
        val_dice_history.append(val_dice)
        loss_seg_history.append(train_loss_seg)
        loss_skel_history.append(train_loss_skel)
        loss_cldice_history.append(train_loss_cldice)
        
        file_training = open(res + "/training.txt", "a")
        file_training.write("loss:" + str(train_loss) + ',loss_seg:' + str(train_loss_seg) + ',loss_skel:' + str(train_loss_skel) + ',loss_cldice:' + str(train_loss_cldice) + ',val_loss:' + str(val_loss) \
                            + ",val_loss_seg:" + str(val_loss_seg) + ",val_loss_skel:" + str(val_loss_skel) + ",val_loss_cldice" + str(val_loss_cldice)
                            + ',val_dice:' + str(val_dice) + ',time:' + str(epoch_duration) + '\n')
        file_training.close()
        
        if scheduler is not None:
            lr = scheduler.get_last_lr()
            lr_history.append(lr)
            scheduler.step()
            
        else:
            lr = [get_lr(optimizer)]
            lr_history.append(lr)
        
        log_dictionnary = {
            "Train Loss": train_loss,
            "Train Loss segmentation": train_loss_seg,
            "Train Loss centerline": train_loss_skel,
            "Train Loss clDice ": train_loss_cldice,
            "Validation Loss": val_loss,
            "Validation Loss segmentation": val_loss_seg,
            "Validation Loss centerline": val_loss_skel,
            "Validation Loss clDice": val_loss_cldice,
            "Validation Dice metric": val_dice,
            "Learning Rate": float(lr[0]),
            "Epoch": 1}
        wandb.log(log_dictionnary)
        
        for t in range(1, epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            indices_train = np.random.choice(np.arange(0, len(dataset_train)), size=train_size_epoch)
            sampler_train = SubsetRandomSampler(indices_train)
            indices_val = np.random.choice(np.arange(0, len(dataset_val)), size=val_size_epoch)
            sampler_val = SubsetRandomSampler(indices_val)
            
            train_data = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers=4, pin_memory=True)
            val_data = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, num_workers=4, pin_memory=True)
            
            if loss_scheduling:
                if t < int(epochs / 2):
                    alpha_dice_ce = (t / epochs) + 0.5
                else:
                    alpha_dice_ce = 1.
        
            logs = train_loop_cascaded(train_data, val_data, model, optimizer, loss_param, alpha, device, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
            
            train_loss = logs["train_loss"]
            train_loss_seg = logs["train_loss_seg"]
            train_loss_skel = logs["train_loss_skel"]
            train_loss_cldice = logs["train_loss_cldice"]
            val_loss = logs["val_loss"]
            val_loss_seg = logs["val_loss_seg"]
            val_loss_skel = logs["val_loss_skel"]
            val_loss_cldice = logs["val_loss_cldice"]
            val_dice = logs["val_dice"]
            epoch_duration = logs["epoch_duration"]
            
            train_history.append(train_loss)
            val_history.append(val_loss)
            val_dice_history.append(val_dice)
            loss_seg_history.append(train_loss_seg)
            loss_skel_history.append(train_loss_skel)
            loss_cldice_history.append(train_loss_cldice)
        
            file_training = open(res + "/training.txt", "a")
            file_training.write("loss:" + str(train_loss) + ',loss_seg:' + str(train_loss_seg) + ',loss_skel:' + str(train_loss_skel) + ',loss_cldice:' + str(train_loss_cldice) + ',val_loss:' + str(val_loss) \
                                + ",val_loss_seg:" + str(val_loss_seg) + ",val_loss_skel:" + str(val_loss_skel) + ",val_loss_cldice" + str(val_loss_cldice)
                                + ',val_dice:' + str(val_dice) + ',time:' + str(epoch_duration) + '\n')
            file_training.close()
            
            if scheduler is not None:
                lr = scheduler.get_last_lr()
                lr_history.append(lr)
                scheduler.step()
                
            else:
                lr = [get_lr(optimizer)]
                lr_history.append(lr)
            
            log_dictionnary = {
                "Train Loss": train_loss,
                "Train Loss segmentation": train_loss_seg,
                "Train Loss centerline": train_loss_skel,
                "Train Loss clDice ": train_loss_cldice,
                "Validation Loss": val_loss,
                "Validation Loss segmentation": val_loss_seg,
                "Validation Loss centerline": val_loss_skel,
                "Validation Loss clDice": val_loss_cldice,
                "Validation Dice metric": val_dice,
                "Learning Rate": float(lr[0]),
                "Epoch": t + 1}
            
            wandb.log(log_dictionnary)
            
            if (val_loss < val_loss_save):
                torch.save(model, res + '/model.pth')
                torch.save(model.state_dict(), res + '/model_weights.pth')
                val_loss_save = val_loss
                epoch_save = t + 1
        
        torch.save(model, res + '/final_model.pth')
        print("Done!")
        return train_history, val_history, val_dice_history, loss_seg_history, loss_skel_history, lr_history, val_loss_save, epoch_save
