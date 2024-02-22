#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:35:04 2021

@author: rouge
"""
import os
import numpy as np
import warnings
import json
import argparse
import matplotlib.pyplot as plt
import wandb
from sklearn.model_selection import train_test_split, KFold

import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary
from torch.optim.lr_scheduler import LinearLR
from torchviz import make_dot

from monai.transforms import LoadImaged, RandScaleIntensityd, RandRotated, RandGaussianSmoothd, RandGaussianNoised, \
    RandAdjustContrastd, RandAxisFlipd, RandZoomd, NormalizeIntensityd, RandSpatialCropd, EnsureChannelFirstd, Compose
from monai.data import CacheDataset


import sys
sys.path.append('..')
from network.unet import My_Unet_tiny
from utils.utils_pytorch import get_lr
from utils.utils_training import train_loop

# This warning will be patch in new versions of monai
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# %% Define parameters for training

# Parameters given by the user
parser = argparse.ArgumentParser(description='3D-Unet for vascular network segmentation')
parser.add_argument('--batch_size', metavar='batch_size', type=int, nargs="?", default=2, help='Batch size for training phase')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, nargs="?", default=0.01, help='Learning rate for training phase')
parser.add_argument('--loss_param', metavar='loss_param', type=str, nargs="?", default='Both', help='Choose loss function')
parser.add_argument('--epochs', metavar='epochs', type=int, nargs="?", default=3, help='Number of epochs for training phase')
parser.add_argument('--input', metavar='input', type=str, nargs="?", default='Both', help='Type of input data')
parser.add_argument('--opt', metavar='opt', type=str, nargs="?", default='SGD', help='Optimizer used during training')
parser.add_argument('--fold', metavar='fold', type=int, nargs="?", default=0, help='Fold to choose')
parser.add_argument('--nbr_batch_epoch', nargs='?', type=int, default=3, help='Number of batch by epoch')
parser.add_argument('--job_name', metavar='job_name', type=str, nargs="?", default='Local', help='Name of job on the cluster')
parser.add_argument('--dir_data', metavar='dir_data', type=str, nargs="?", default='../data', help='Data\'s directory')
parser.add_argument('--features', nargs='+', type=int, default=[2, 2, 2, 2, 2, 2], help='Number of features for each layer in the decoder')
parser.add_argument('--patch_size', nargs='+', type=int, default=[64, 64, 64], help='Patch _size')
parser.add_argument("--scheduler", help="Set learning rate scheduler for training", action="store_true")
parser.add_argument("--nesterov", help="Use SGD with nesterov momentum", action="store_true")
parser.add_argument('--entity', metavar='entity', type=str, default='pierre-rouge', help='Entity for W&B')
args = parser.parse_args()

# Check if GPU is available for training
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


# Save parameters for training and ceate res directories
dir_res = '../res/cascaded_unet'
if not os.path.exists(dir_res + '/skeletonization'):
    os.makedirs(dir_res + '/skeletonization')
num = 0
for f in os.listdir(dir_res + '/skeletonization'):
    num += 1
num += 1

res = dir_res + '/skeletonization/' + args.job_name + '_' + str(num)
dir_exist = 0
while dir_exist != 1:
    if os.path.exists(res):
        num += 1
        res = dir_res + '/skeletonization/' + args.job_name + '_' + str(num)
    if not os.path.exists(res):
        os.makedirs(res)
        dir_exist = 1
           
# Set variables for parameteres
batch_size = args.batch_size
learning_rate = args.learning_rate
loss_param = args.loss_param
epochs = args.epochs
input_ = args.input
opt = args.opt
job_name = args.job_name
dir_data = args.dir_data
fold_ = args.fold
features = tuple(args.features)
patch_size = tuple(args.patch_size)
nbr_batch_epoch = args.nbr_batch_epoch
nesterov = args.nesterov
scheduler = args.scheduler
size_train_epoch = nbr_batch_epoch * batch_size
size_val_epoch = 50 * batch_size
val_size_epoch = 1.
weight_decay = 3 * 10 ** -5

    
wandb_config = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "optimizer": opt,
    "job_name": job_name,
    "fold": fold_,
    "input": input_,
    "loss": loss_param,
    "features": features,
    "patch_size": patch_size,
    "nbr_batch_epoch": nbr_batch_epoch,
    "nesterov": nesterov,
    "val_size_epoch": val_size_epoch,
    "weigth_decay": weight_decay,
    "scheduler": scheduler,
    "value_momentum": 0.99}

# Save config in json file
with open(res + '/config_training.json', 'w') as outfile:
    json.dump(wandb_config, outfile)

# Init weight and biases
if job_name == 'Local':
    wandb.init(project='Debug', entity=args.entity, config=wandb_config, name=job_name)
else:
    wandb.init(project='cascaded-unet-skeletonization', entity=args.entity, config=wandb_config, name=job_name)


# %% Data splitting

# Select data's directories
dir_inputs = os.path.join(dir_data, 'Images')
dir_GT = os.path.join(dir_data, 'Skeletons')

# Segmentation as input to perform skeletonization
dir_inputs_seg = os.path.join(dir_data, 'GT')

# Separate patients for training, validation and test
patient = []
for (root, directory, file) in os.walk(dir_inputs):
    for f in file:
        split = f.split('-')
        patient.append(split[0])
patient = np.array(patient)

k = 5
sp = KFold(n_splits=k, shuffle=True, random_state=42)
for fold, (train, test) in enumerate(sp.split(np.arange(len(patient)))):
    if fold == fold_:
        idx_train_temp = train
        idx_test = test

patient_train = list(patient[idx_train_temp])
patient_test = list(patient[idx_test])

# Create dictionnaries for the pytorch dataset
data_train = []
data_test = []
for (root, directory, file) in os.walk(dir_inputs):
    for f in file:
        split = f.split('-')
        name = f.split('.')[0]
        if split[0] in patient_train:
            data_train.append(dict(zip(['image', 'segmentation', 'GT'], [dir_inputs + '/' + f, dir_inputs_seg + '/' + name + '_GT.nii.gz', dir_GT + '/' + name + '_GT_skeleton.nii.gz'])))
            
for (root, directory, file) in os.walk(dir_inputs):
    for f in file:
        split = f.split('-')
        name = f.split('.')[0]
        if split[0] in patient_test:
            data_test.append(dict(zip(['image', 'segmentation', 'GT', 'filename'], [dir_inputs + '/' + f, dir_inputs_seg + '/' + name + '_GT.nii.gz', dir_GT + '/' + name + '_GT_skeleton.nii.gz', f])))
            
            
# Save patient split in json file
with open(res + '/config_training.json') as json_file:
    data = json.load(json_file)
    data['train_set'] = patient_train
    data['test_set'] = patient_test
    
with open(res + '/config_training.json', 'w') as outfile:
    json.dump(data, outfile)
    
# Define transforms
if input_ == 'MRI':
    keys = ('image', 'GT')
elif input_ == 'segmentation':
    keys = ('segmentation', 'GT')
elif input_ == 'Both':
    keys = ('image', 'segmentation', 'GT')
    
range_rotation = (-0.523, 0.523)
prob = 0.2

if input_ == 'MRI' or input_ == 'Both':
    transform_io = [LoadImaged(keys), EnsureChannelFirstd(keys),
                    NormalizeIntensityd(keys=('image')), RandSpatialCropd(keys, roi_size=patch_size, random_size=False)]
    
    transform_augmentation = [RandRotated(keys, prob=prob, range_x=range_rotation, range_y=range_rotation, range_z=range_rotation), 
                              RandZoomd(keys, prob=prob, min_zoom=0.7, max_zoom=1.4),  RandGaussianNoised(keys=('image'), prob=prob),
                              RandGaussianSmoothd(keys=('image'), prob=prob), RandScaleIntensityd(keys=('image'), factors=0.3, prob=prob), RandAxisFlipd(keys, prob=prob), 
                              RandAdjustContrastd(keys=('image'), prob=prob)]
    
    
elif input_ == 'segmentation':
     transform_io = [LoadImaged(keys), EnsureChannelFirstd(keys), RandSpatialCropd(keys, roi_size=patch_size, random_size=False)]   

     transform_augmentation = [RandRotated(keys, prob=prob, range_x=range_rotation, range_y=range_rotation, range_z=range_rotation), 
                              RandZoomd(keys, prob=prob, min_zoom=0.7, max_zoom=1.4),  RandAxisFlipd(keys, prob=prob)]
   
transform_train = Compose(transform_io + transform_augmentation)


transform = Compose(transform_io)


# Create dataset and dataloader~
dataset_train = CacheDataset(data_train, transform_train)
dataset_val = CacheDataset(data_test, transform)

indices_train = np.random.choice(np.arange(0, len(dataset_train)), size=size_train_epoch)
sampler_train = SubsetRandomSampler(indices_train)
indices_val = np.random.choice(np.arange(0, len(dataset_val)), size=size_val_epoch)
sampler_val = SubsetRandomSampler(indices_val)

train_data = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers=4)
val_data = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, num_workers=4)

#%% Define Model and save the summary
if input_ == "Both":
    in_channels = 2
else:
    in_channels = 1
    

kernel_size = (3, 3, 3, 3)
strides = (1, 2, 2, 2)
features = features[:4]
model = My_Unet_tiny(dim=3, in_channel=in_channels, features=features, strides=strides, kernel_size=kernel_size)
    
sigmoid = nn.Sigmoid()
model = model.float()
model = model.to(device)
x = torch.zeros(1, in_channels, patch_size[0], patch_size[1], patch_size[2], dtype=torch.float, requires_grad=False, device=device)
y = model(x)
dot = make_dot(y)
dot.format = 'jpg'
dot.render(res + "/architecture")
model_summary = str(summary(model, input_size=(batch_size, in_channels, patch_size[0], patch_size[1], patch_size[2]), dtypes=[torch.float]))
file_summary = open(res + "/model_summary.txt", "a")
file_summary.write(model_summary)
file_summary.close()

if opt == "Adam":
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif opt == "SGD":
    if nesterov:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.99, nesterov=True)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

if scheduler:
    print("Learning rate scheduler activated")
    sch = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
else:
    print("No learning rate scheduler")
    sch = None

#%%Training

train_history = []
train_dice_history = []
train_cldice_history = []
val_history = []
val_dice_history = []

print(f"Epoch {1}\n-------------------------------")
epoch_save = 1
logs = train_loop(train_data, val_data, model=model, loss_param=loss_param, input_=input_, optimizer=optimizer, device=device, epoch=1, max_epoch=epochs)

if loss_param == 'clDice':
    loss = logs["train_loss"]
    dice_loss = logs['train_dice_loss']
    cldice_loss = logs["train_cldice_loss"]
    val_loss = logs["val_loss"]
    val_dice = logs["val_dice"]
    epoch_duration = logs["epoch_duration"]
    
    val_loss_save = val_loss
    
    train_history.append(loss)
    train_dice_history.append(dice_loss)
    train_cldice_history.append(cldice_loss)
    val_history.append(val_loss)
    val_dice_history.append(val_dice)
    
    file_training = open(res + "/training.txt", "a")
    file_training.write("loss:" + str(loss) + ',dice_loss:' + str(dice_loss) + ',cldice_loss:' + str(cldice_loss)+ ',val_loss:' + str(val_loss) + ',val_dice:' + str(val_dice) + ',time:' + str(epoch_duration) + '\n')
    file_training.close()
    
    if sch is not None:
        lr = sch.get_last_lr()
        sch.step()
        
    else:
        lr = [get_lr(optimizer)]
    
    log_dictionnary = {
        "Loss": loss,
        "Dice Loss": dice_loss,
        "clDice Loss": cldice_loss,
        "Validation Loss": val_loss,
        "Validation Dice metric": val_dice,
        "Learning Rate": float(lr[0]),
        "Epoch": 1}
    wandb.log(log_dictionnary)
else:
    loss = logs["train_loss"]
    val_loss = logs["val_loss"]
    val_dice = logs["val_dice"]
    epoch_duration = logs["epoch_duration"]
    
    val_loss_save = val_loss
    
    train_history.append(loss)
    val_history.append(val_loss)
    val_dice_history.append(val_dice)
    
    file_training = open(res + "/training.txt", "a")
    file_training.write("loss:" + str(loss) + ',val_loss:' + str(val_loss) + ',val_dice:' + str(val_dice) + ',time:' + str(epoch_duration) + '\n')
    file_training.close()
    
    if sch is not None:
        lr = sch.get_last_lr()
        sch.step()
        
    else:
        lr = [get_lr(optimizer)]
    
    log_dictionnary = {
        "Loss": loss,
        "Validation Loss": val_loss,
        "Validation Dice metric": val_dice,
        "Learning Rate": float(lr[0]),
        "Epoch": 1}
    wandb.log(log_dictionnary)

for t in range(1, epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    
    indices_train = np.random.choice(np.arange(0, len(dataset_train)), size=size_train_epoch)
    sampler_train = SubsetRandomSampler(indices_train)
    indices_val = np.random.choice(np.arange(0, len(dataset_val)), size=size_val_epoch)
    sampler_val = SubsetRandomSampler(indices_val)

    
   
    train_data = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers=4)
    val_data = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, num_workers=4)

    logs = train_loop(train_data, val_data, model=model, loss_param=loss_param, input_=input_, optimizer=optimizer, device=device, epoch=1, max_epoch=epochs)
    if loss_param == 'clDice':
        loss = logs["train_loss"]
        dice_loss = logs['train_dice_loss']
        cldice_loss = logs["train_cldice_loss"]
        val_loss = logs["val_loss"]
        val_dice = logs["val_dice"]
        epoch_duration = logs["epoch_duration"]
        
        train_history.append(loss)
        train_dice_history.append(dice_loss)
        train_cldice_history.append(cldice_loss)
        val_history.append(val_loss)
        val_dice_history.append(val_dice)
        
        file_training = open(res + "/training.txt", "a")
        file_training.write("loss:" + str(loss) + ',dice_loss:' + str(dice_loss) + ',cldice_loss:' + str(cldice_loss)+ ',val_loss:' + str(val_loss) + ',val_dice:' + str(val_dice) + ',time:' + str(epoch_duration) + '\n')
        file_training.close()
        
        if sch is not None:
            lr = sch.get_last_lr()
            sch.step()
            
        else:
            lr = [get_lr(optimizer)]
        
        log_dictionnary = {
            "Loss": loss,
            "Dice Loss": dice_loss,
            "clDice Loss": cldice_loss,
            "Validation Loss": val_loss,
            "Validation Dice metric": val_dice,
            "Learning Rate": float(lr[0]),
            "Epoch": 1}
        wandb.log(log_dictionnary)
    else:
        loss = logs["train_loss"]
        val_loss = logs["val_loss"]
        val_dice = logs["val_dice"]
        epoch_duration = logs["epoch_duration"]
        
        train_history.append(loss)
        val_history.append(val_loss)
        val_dice_history.append(val_dice)
        
        file_training = open(res + "/training.txt", "a")
        file_training.write("loss:" + str(loss) + ',val_loss:' + str(val_loss) + ',val_dice:' + str(val_dice) + ',time:' + str(epoch_duration) + '\n')
        file_training.close()
        
        if sch is not None:
            lr = sch.get_last_lr()
            sch.step()
            
        else:
            lr = [get_lr(optimizer)]
        
        log_dictionnary = {
            "Loss": loss,
            "Validation Loss": val_loss,
            "Validation Dice metric": val_dice,
            "Learning Rate": float(lr[0]),
            "Epoch": 1}
        wandb.log(log_dictionnary)
    
    # Save best model
    if (val_loss < val_loss_save):
        val_loss_save = val_loss
        epoch_save = t + 1
    
torch.save(model, res + '/final_model.pth')
print("Done!")

file_training = open(res + "/training_n°" + str(num) + ".txt", "a")
file_training.write("Epoch of best model :" + str(epoch_save) + '\n')
file_training.close()

# Save training curves
abscisse = np.arange(1, epochs + 1, 1)
# Save figures
abscisse = np.arange(1, epochs + 1, 1)
color = 'blue'
fig, ax1 = plt.subplots()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss value' + '(' + 'custom' + ')', color=color)
ax1.plot(abscisse, train_history, color='blue', label='Loss Train')
ax1.plot(abscisse, val_history, color='orange', label='Loss Val')
if loss_param == 'clDice':
    ax1.plot(abscisse, train_dice_history, color='red', label='Dice Loss Train')
    ax1.plot(abscisse, train_cldice_history, color='yellow', label='clDice Loss Val')
ax1.scatter(epoch_save, val_loss_save, color='red', marker='*', label='Model saved')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.15))

ax2 = ax1.twinx()
color = 'green'
ax2.set_ylabel('DICE score', color=color)
ax2.plot(abscisse, val_dice_history, color=color, label='DICE val')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1.1))

plt.title('Curves_training_n°' + str(num))
plt.savefig(res + "/curves_training_n°" + str(num) + ".png")


wandb.finish()
