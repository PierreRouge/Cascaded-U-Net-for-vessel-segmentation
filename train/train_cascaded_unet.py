#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:27:07 2022

@author: rouge
"""

import argparse
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import wandb
import warnings
from sklearn.model_selection import train_test_split, KFold

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.optim.lr_scheduler import LinearLR
from torchviz import make_dot

from monai.transforms import AddChanneld, LoadImaged, EnsureTyped, NormalizeIntensityd, RandSpatialCropd, adaptor, SqueezeDimd, CopyItemsd, Compose, EnsureChannelFirstd
from monai.data import CacheDataset

from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform

import sys
sys.path.append('..')
from utils.utils_training import training_cascaded
from utils.utils_monai import skeletonized
from network.cascaded_unet import Cascaded_Unet_tiny, Cascaded_Unet_tiny_pretrained

# This warning will be patch in new versions of monai
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Parameters given by the user
parser = argparse.ArgumentParser(description='Multitask network for vascular network segmentation')
parser.add_argument('--batch_size', metavar='batch_size', type=int, nargs="?", default=2, help='Batch size for training phase')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, nargs="?", default=0.01, help='Learning rate for training phase')
parser.add_argument('--loss_param', metavar='loss_param', type=str, nargs="?", default='clDice', help='hoose loss function')
parser.add_argument('--epochs', metavar='epochs', type=int, nargs="?", default=3, help='Number of epochs for training phase')
parser.add_argument('--opt', metavar='opt', type=str, nargs="?", default='SGD', help='Optimizer used during training')
parser.add_argument('--fold', metavar='fold', type=int, nargs="?", default=0, help='Fold to choose')
parser.add_argument('--lambda1', metavar='lambda1', type=float, nargs="?", default=1.0, help='Weight of segmentation loss')
parser.add_argument('--lambda2', metavar='lambda2', type=float, nargs="?", default=1.0, help='Weight of centerline loss')
parser.add_argument('--lambda3', metavar='lambda3', type=float, nargs="?", default=0.5, help='Weight of clDice loss')
parser.add_argument('--nbr_batch_epoch', nargs='?', type=int, default=3, help='Number of batch by epoch')
parser.add_argument('--job_name', metavar='job_name', type=str, nargs="?", default='Local', help='Name of job on the cluster')
parser.add_argument('--dir_data', metavar='dir_data', type=str, nargs="?", default='../data/', help='Data\'s directory')
parser.add_argument('--dir_weights_segmentation', metavar='dir_data', type=str, nargs="?", default='../pretrained_weights/segmentation/', help='Weight\'s directory')
parser.add_argument('--dir_weights_skeletonization', metavar='dir_data', type=str, nargs="?", default='../pretrained_weights/skeletonization/', help='Weight\'s directory')
parser.add_argument("--pretrained", help="Use pretrained version", action="store_true")
parser.add_argument("--freeze_skeleton", help="Freeze network for skeletization", action="store_true")
parser.add_argument('--features', nargs='+', type=int, default=[2, 4, 4, 4, 4, 4], help='Number of features for each layer in the decoder')
parser.add_argument('--patch_size', nargs='+', type=int, default=[64, 64, 64], help='Patch _size')
parser.add_argument("--scheduler", help="Set learning rate scheduler for training", action="store_true")
parser.add_argument("--nesterov", help="Use SGD with nesterov momentum", action="store_true")
parser.add_argument('--entity', metavar='entity', type=str, default='', help='Entity for W&B')
args = parser.parse_args()

# Check for GPU
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# Create directories to store parameters and results
dir_res = '../res/cascaded_unet/cascaded_unet'
if not os.path.exists(dir_res):
    os.makedirs(dir_res)

num = 0
for f in os.listdir(dir_res):
    num += 1
num += 1

res = dir_res + '/training_n°' + str(num)
dir_exist = 0
while dir_exist != 1:
    if os.path.exists(res):
        num += 1
        res = dir_res + '/training_n°' + str(num)
    if not os.path.exists(res):
        os.makedirs(res)
        dir_exist = 1
        
# %% Set parameters for training

# Set variables for parameteres
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
opt = args.opt
job_name = args.job_name
dir_data = args.dir_data
dir_weights_segmentation = args.dir_weights_segmentation
dir_weights_skeletonization = args.dir_weights_skeletonization
fold_ = args.fold
loss_param = args.loss_param
features = tuple(args.features)
patch_size = tuple(args.patch_size)
nbr_batch_epoch = args.nbr_batch_epoch
pretrained = args.pretrained
freeze_skeleton = args.freeze_skeleton
nesterov = args.nesterov
scheduler = args.scheduler
size_train_epoch = nbr_batch_epoch * batch_size
size_val_epoch = 50 * batch_size
weight_decay = 3 * 10 ** -5
dim = 3
in_channel = 1
lambda1 = args.lambda1
lambda2 = args.lambda2
lambda3 = args.lambda3
sigmoid = nn.Sigmoid()

wandb_config = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "optimizer": opt,
    "job_name": job_name,
    "dir_data": dir_data,
    "fold": fold_,
    "loss": loss_param,
    "features": features,
    "patch_size": patch_size,
    "nbr_batch_epoch": nbr_batch_epoch,
    "freeze_skeleton": freeze_skeleton,
    "nesterov": nesterov,
    "size_val_epoch": size_val_epoch,
    "weigth_decay": weight_decay,
    "scheduler": scheduler,
    "value_momentum": 0.9,
    "lambda1": lambda1,
    "lambda2": lambda2,
    "lambda3": lambda3}
 
with open(res + '/config_training.json', 'w') as outfile:
    json.dump(wandb_config, outfile)


if job_name == 'Local':
    wandb.init(project='Debug', entity=args.entity, config=wandb_config, name=job_name)
else:
    wandb.init(project='cascaded_unet', entity=args.entity, config=wandb_config, name=job_name)
    
# %% Data splitting

# Select data's directories
dir_inputs = os.path.join(dir_data, 'Images')
dir_GT = os.path.join(dir_data, 'GT')
dir_skel = dir_data + os.path.join(dir_data, 'Skeletons')
    
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
        
idx_train, idx_val = train_test_split(idx_train_temp, test_size=0.05, shuffle=True, random_state=42)

patient_train = list(patient[idx_train_temp])
patient_val = list(patient[idx_val])
patient_test = list(patient[idx_test])

        
patient_val = patient_test

data_train = []
data_val = []
data_test = []
for (root, directory, file) in os.walk(dir_inputs):
    for f in file:
        split = f.split('-')
        if split[0] in patient_train:
            data_train.append(dict(zip(['image', 'GT'], [dir_inputs + '/' + f, dir_GT + '/' + f[:-7] + '_GT.nii.gz'])))
            
for (root, directory, file) in os.walk(dir_inputs):
    for f in file:
        split = f.split('-')
        if split[0] in patient_val:
            data_val.append(dict(zip(['image', 'GT', 'filename'], [dir_inputs + '/' + f, dir_GT + '/' + f[:-7] + '_GT.nii.gz', f])))
            
with open(res + '/config_training.json') as json_file:
    data = json.load(json_file)
    data['train_set'] = patient_train
    data['validation_set'] = patient_val
    data['test_set'] = patient_test
    
with open(res + '/config_training.json', 'w') as outfile:
    json.dump(data, outfile)
    

# Define transforms
keys = ('image', 'GT')
range_rotation = (-0.523, 0.523)
outputs = {'image': 'image', 'GT': 'GT'}
prob = 0.2


transform_io = [LoadImaged(keys), EnsureChannelFirstd(keys), NormalizeIntensityd(keys='image'), RandSpatialCropd(keys, roi_size=patch_size, random_size=False)]

# Data augmentation
transform_augmentation = [AddChanneld(keys), EnsureTyped(keys, data_type="numpy")]

transform_augmentation.append(adaptor(SpatialTransform(patch_size, patch_center_dist_from_border=None,
                                                       do_elastic_deform=False, alpha=(0.0, 900.0),
                                                       sigma=(9.0, 13.0),
                                                       do_rotation=True, angle_x=range_rotation, angle_y=range_rotation,
                                                       angle_z=range_rotation, p_rot_per_axis=1,
                                                       do_scale=True, scale=(0.7, 1.4),
                                                       border_mode_data="constant", border_cval_data=0, order_data=3,
                                                       border_mode_seg="constant", border_cval_seg=-1,
                                                       order_seg=1, random_crop=False, p_el_per_sample=0.2,
                                                       p_scale_per_sample=0.2, p_rot_per_sample=0.2,
                                                       independent_scale_for_each_axis=False, data_key='image', label_key=('GT')), outputs))

transform_augmentation.append(adaptor(GaussianNoiseTransform(p_per_sample=0.1, data_key='image'), outputs))

transform_augmentation.append(adaptor(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5, data_key='image'), outputs))

transform_augmentation.append(adaptor(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15, data_key='image'), outputs))

transform_augmentation.append(adaptor(ContrastAugmentationTransform(p_per_sample=0.15, data_key='image'), outputs))

transform_augmentation.append(adaptor(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                                     p_per_channel=0.5,
                                                                     order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                                     ignore_axes=None, data_key='image'), outputs))

transform_augmentation.append(adaptor(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1, data_key='image'), outputs))  # inverted gamma

transform_augmentation.append(adaptor(MirrorTransform((0, 1, 2), data_key='image', label_key=('GT')), outputs))

transform_train = Compose(transform_io + transform_augmentation + [CopyItemsd(keys='GT', times=1, names=['skeleton']), skeletonized(keys='skeleton', datatype='numpy'), SqueezeDimd(keys=('image', 'GT', 'skeleton'), dim=0)])
            
    
transform = Compose(transform_io + [CopyItemsd(keys='GT', times=1, names=['skeleton']), skeletonized(keys='skeleton')])

# Create dataset and dataloader~
dataset_train = CacheDataset(data_train, transform_train)
dataset_val = CacheDataset(data_val, transform)
dataset_test = CacheDataset(data_test, transform)

test_data = DataLoader(dataset_test, batch_size=batch_size, num_workers=4)

# %% Define model
kernel_size = (3, 3, 3, 3)
strides = (1, 2, 2, 2)
features = features[:4]

# If pretrained weights
if pretrained:
    path_weights_segmentation = dir_weights_segmentation + "fold_" + str(args.fold) + ".pth"
    path_weights_skeleton = dir_weights_skeletonization + "fold_" + str(args.fold) + ".pth"
    
    in_channel1 = 1
    in_channel2 = 2
    
    weights_segmentation = torch.load(path_weights_segmentation).state_dict()
    weights_skeleton = torch.load(path_weights_skeleton).state_dict()
    
    model = Cascaded_Unet_tiny_pretrained(dim, in_channel1, in_channel2, features, strides, kernel_size, weights_segmentation, weights_skeleton, freeze_skeleton)

else:
    model = Cascaded_Unet_tiny(dim, in_channel, features, strides, kernel_size)
    

model = model.float()
model = model.to(device)
x = torch.zeros(1, 1, patch_size[0], patch_size[1], patch_size[2], dtype=torch.float, requires_grad=False, device=device)
y = model(x)
dot = make_dot(y)
dot.format = 'jpg'
dot.render(res + "/architecture")
model_summary = str(summary(model, input_size=(batch_size, 1, patch_size[0], patch_size[1], patch_size[2]), dtypes=[torch.float]))
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

# %%Training

train_history, val_history, dice_val_history, loss_seg_history, loss_skel_history, lr_history, val_loss_save, epoch_save = training_cascaded(dataset_train, dataset_val, model, optimizer, epochs, batch_size, device, res, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, loss_param=loss_param, train_size_epoch=size_train_epoch, val_size_epoch=size_val_epoch)


# %% Save training and figures

file_training = open(res + "/training_n°" + str(num) + ".txt", "a")
file_training.write("Epoch of best model :" + str(epoch_save) + '\n')
file_training.close()


# Save figures
abscisse = np.arange(1, epochs + 1, 1)
color = 'blue'
fig, ax1 = plt.subplots()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss value' + '(' + 'custom' + ')', color=color)
ax1.plot(abscisse, train_history, color='blue', label='Loss Train')
ax1.plot(abscisse, loss_seg_history, color='red', label='Loss Seg')
ax1.plot(abscisse, loss_skel_history, color='yellow', label='Loss Skel')
ax1.plot(abscisse, val_history, color='orange', label='Loss Val')
ax1.scatter(epoch_save, val_loss_save, color='red', marker='*', label='Model saved')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.15))

ax2 = ax1.twinx()
color = 'green'
ax2.set_ylabel('DICE score', color=color)
ax2.plot(abscisse, dice_val_history, color=color, label='DICE val')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1.1))

plt.title('Curves_training_n°' + str(num))
plt.savefig(res + "/curves_training_n°" + str(num) + ".png")

fig2, ax1 = plt.subplots()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Learning Rate')
ax1.plot(abscisse, lr_history, color='blue', label='Learning Rate')
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.15))
plt.title('LR_training_n°' + str(num))
plt.savefig(res + "/LR_training_n°" + str(num) + ".png")

wandb.finish()
