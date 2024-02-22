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

from monai.transforms import AddChanneld, LoadImaged, EnsureTyped, NormalizeIntensityd, RandSpatialCropd, adaptor, SqueezeDimd, Compose, EnsureChannelFirstd
from monai.data import CacheDataset

from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform

import sys
sys.path.append('..')
from utils.utils_pytorch import get_lr
from utils.utils_training import train_loop
from network.deepvesselnet import DeepVesselNet

# This warning will be patch in new versions of monai
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# %% Define parameters for training

# Parameters given by the user
parser = argparse.ArgumentParser(description='DeepVesselNet for vascular network segmentation')
parser.add_argument('--batch_size', metavar='batch_size', type=int, nargs="?", default=2, help='Batch size for training phase')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, nargs="?", default=0.01, help='Learning rate for training phase')
parser.add_argument('--loss_param', metavar='loss_param', type=str, nargs="?", default='Dice', help='Choose loss function')
parser.add_argument('--alpha', metavar='alpha', type=float, nargs="?", default=0.5, help='Weight for Dice loss in Dice + clDice loss')
parser.add_argument('--epochs', metavar='epochs', type=int, nargs="?", default=10, help='Number of epochs for training phase')
parser.add_argument('--opt', metavar='opt', type=str, nargs="?", default='SGD', help='Optimizer used during training')
parser.add_argument('--fold', metavar='fold', type=int, nargs="?", default=0, help='Fold to choose')
parser.add_argument('--nbr_batch_epoch', nargs='?', type=int, default=50, help='Number of batch by epoch')
parser.add_argument('--job_name', metavar='job_name', type=str, nargs="?", default='Local', help='Name of job on the cluster')
parser.add_argument('--dir_data', metavar='dir_data', type=str, nargs="?", default='../data', help='Data directory')
parser.add_argument('--features', nargs='+', type=int, default=[5, 10, 20, 50], help='Number of features for each layer in the decoder')
parser.add_argument('--patch_size', nargs='+', type=int, default=[32, 32, 32], help='Patch _size')
parser.add_argument("--scheduler", help="Set learning rate scheduler for training", action="store_true")
parser.add_argument("--nesterov", help="Use SGD with nesterov momentum", action="store_true")
parser.add_argument('--entity', metavar='entity', type=str, default='', help='Entity for W&B')
args = parser.parse_args()

# Check if GPU is available for training
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


# Save parameters for training and ceate res directories
dir_res = '../res/deepvesselnet'
if not os.path.exists(dir_res + '/deepvesselnet'):
    os.makedirs(dir_res + '/deepvesselnet')
num = 0
for f in os.listdir(dir_res + '/deepvesselnet'):
    num += 1
num += 1

res = dir_res + '/deepvesselnet/' + args.job_name + '_' + str(num)
dir_exist = 0
while dir_exist != 1:
    if os.path.exists(res):
        num += 1
        res = dir_res + '/deepvesselnet/' + args.job_name + '_' + str(num)
    if not os.path.exists(res):
        os.makedirs(res)
        dir_exist = 1
           
# Set variables for parameteres
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
opt = args.opt
job_name = args.job_name
dir_data = args.dir_data
fold_ = args.fold
loss_param = args.loss_param
alpha = args.alpha
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
    "loss": loss_param,
    "alpha": alpha,
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
    wandb.init(project='cascaded-unet-segmentation', entity=args.entity, config=wandb_config, name=job_name)


# %% Data splitting

# Select data's directories
dir_inputs = os.path.join(dir_data, 'Images')
dir_GT = os.path.join(dir_data, 'GT')

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
            data_train.append(dict(zip(['image', 'GT'], [dir_inputs + '/' + f, dir_GT + '/' + name + '_GT.nii.gz'])))
            
for (root, directory, file) in os.walk(dir_inputs):
    for f in file:
        split = f.split('-')
        name = f.split('.')[0]
        if split[0] in patient_test:
            data_test.append(dict(zip(['image', 'GT', 'filename'], [dir_inputs + '/' + f, dir_GT + '/' + name + '_GT.nii.gz', f])))
            

# Save patient split in json file
with open(res + '/config_training.json') as json_file:
    data = json.load(json_file)
    data['train_set'] = patient_train
    data['test_set'] = patient_test
    
with open(res + '/config_training.json', 'w') as outfile:
    json.dump(data, outfile)
    
    
# Define transforms
keys = ('image', 'GT')
outputs = {'image': 'image', 'GT': 'GT'}
range_rotation = (-0.523, 0.523)
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


transform_train = Compose(transform_io + transform_augmentation + [SqueezeDimd(keys, dim=0)])
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


# %% Define Model and save the summary

kernel_size = (3, 5, 5, 3)
strides = (1, 1, 1, 1)
features = features[:4]
model = DeepVesselNet(dim=3, in_channel=1, features=features, strides=strides, kernel_size=kernel_size)
    
sigmoid = nn.Sigmoid()
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

train_history = []
val_history = []
val_dice_history = []

print(f"Epoch {1}\n-------------------------------")
epoch_save = 1
logs = train_loop(train_data, val_data, model=model, loss_param=loss_param, input_='MRI', optimizer=optimizer, device=device, epoch=1, max_epoch=epochs, alpha_=alpha)

loss = logs['train_loss']
val_loss = logs['val_loss']
val_dice = logs['val_dice']
epoch_duration = logs['epoch_duration']


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

logs["Learning Rate"] = float(lr[0])
logs["Epoch"] = 1
wandb.log(logs)

for t in range(1, epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    
    indices_train = np.random.choice(np.arange(0, len(dataset_train)), size=size_train_epoch)
    sampler_train = SubsetRandomSampler(indices_train)
    indices_val = np.random.choice(np.arange(0, len(dataset_val)), size=size_val_epoch)
    sampler_val = SubsetRandomSampler(indices_val)

    train_data = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers=4)
    val_data = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, num_workers=4)

    logs = train_loop(train_data, val_data, model=model, loss_param=loss_param, input_='MRI', optimizer=optimizer, device=device, epoch=t + 1, max_epoch=epochs, alpha_=alpha)
    
    loss = logs['train_loss']
    val_loss = logs['val_loss']
    val_dice = logs['val_dice']
    epoch_duration = logs['epoch_duration']
    
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

    logs["Learning Rate"] = float(lr[0])
    logs["Epoch"] = t + 1
    wandb.log(logs)
    
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
