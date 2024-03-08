#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 17:38:53 2022

@author: rouge
"""

# %% Import
import os
import json
import torch
import csv
from torch import nn
import nibabel as nib
import numpy as np
import sys
import argparse
import warnings

from skimage.morphology import ball
from scipy.ndimage.filters import gaussian_filter

from torchvision import transforms
from torch.utils.data import DataLoader

from monai.data import CacheDataset
from monai.transforms import LoadImaged, ToTensord, NormalizeIntensityd, Flipd, Flip, EnsureChannelFirstd, RemoveSmallObjects
from monai.inferers import sliding_window_inference
from monai.metrics import SurfaceDiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric, DiceMetric

sys.path.append('..')
from utils.utils_measure import dice_numpy, cldice_numpy, sensitivity_specificity_precision, mcc_numpy, euler_number_error_numpy, b0_error_numpy, b1_error_numpy, b2_error_numpy

# This warning will be patch in new versions of monai
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# %% Define model, data and outputs directories

parser = argparse.ArgumentParser(description='Inference for segmentation')
parser.add_argument('--dir_training', metavar='dir_training', type=str, nargs="?", default='/home/rouge/Documents/git/Cascaded-U-Net-for-vessel-segmentation/res/deep_distance/saved_for_inference/Unet_ddt_fold0_Bullitt_18', help='Training directory')
parser.add_argument('--dir_data', metavar='dir_data', type=str, nargs="?", default='/home/rouge/Documents/Thèse_Rougé_Pierre/Data/Bullit/raw/', help='Data directory')
parser.add_argument('--K', metavar='K', type=int, nargs="?", default=8, help='Number of class for distance map')
parser.add_argument('--patch_size', nargs='+', type=int, default=[192, 192, 64], help='Patch _size')
parser.add_argument("--augmentation", default=False, help="Do test time augmentation", action="store_true")
parser.add_argument("--postprocessing", default=True, help="Do postprocessing", action="store_true")
args = parser.parse_args()

dir_inputs = args.dir_data + 'Images'
dir_res = args.dir_training + '/res'
dir_seg = dir_res + '/seg'
dir_GT = args.dir_data + 'GT'

if not os.path.exists(dir_res):
    os.makedirs(dir_res)
if not os.path.exists(dir_seg):
    os.makedirs(dir_seg)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# %% Define parameters for inference

# Define transform to perform data augmentation during inference
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)
flip1 = Flip(spatial_axis=0)
flip2 = Flip(spatial_axis=1)
flip3 = Flip(spatial_axis=2)

with open(args.dir_training + '/config_training.json') as params:
    params_dict = json.load(params)
    patient_test = params_dict['test_set']


data_test = []
for (root, directory, file) in os.walk(dir_inputs):
    for f in file:
        split = f.split('-')
        name = f.split('.')[0]
        if split[0] in patient_test:
            image = nib.load(dir_inputs + '/' + f)
            headers = image.header
            pixdim = headers['pixdim'][1:4]
            data_test.append(dict(zip(['image1', 'image2', 'image3', 'image4', 'GT', 'filename', 'pixdim'], [dir_inputs + '/' + f, dir_inputs + '/' + f, dir_inputs + '/' + f, dir_inputs + '/' + f, dir_GT + '/' + name + '_GT.nii.gz', f, pixdim])))

# Define transforms
keys = ('image1', 'image2', 'image3', 'image4', 'GT')
keys2 = ('image1', 'image2', 'image3', 'image4')


transform = transforms.Compose([LoadImaged(keys), ToTensord(keys, dtype=torch.double, track_meta=True), EnsureChannelFirstd(keys), NormalizeIntensityd(keys=keys2),
                                Flipd(keys=('image2'), spatial_axis=0), Flipd(keys=('image3'), spatial_axis=1), Flipd(keys=('image4'), spatial_axis=2)])

# Create dataset and dataloader
dataset_test = CacheDataset(data_test, transform)

test_data = DataLoader(dataset_test, batch_size=1, num_workers=4)

# %% Peform inference


# GEOMETRY-AWARE REFINEMENT

def gaussian_map_ddt(patch_size, zu):
    tmp = np.ones(patch_size)
    sigma = (zu / 3)
    gaussian_importance_map = gaussian_filter(tmp, sigma, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    return gaussian_importance_map


print('GEOMETRY-AWARE REFINEMENT: preparing soften ball')
channel_dim = 2
list_kernel = []
str_ = channel_dim - 1
k = args.K
for radius in range(1, k):
    print(radius)
    kernel = torch.as_tensor(np.repeat(np.expand_dims(ball(radius), 0)[np.newaxis, ...], str_, axis=0),
                             dtype=torch.float16).to(device)
    gaussian_gar = torch.as_tensor(
        np.repeat(np.expand_dims(gaussian_map_ddt(kernel[0, 0].size(), radius), 0)[np.newaxis, ...], str_, axis=0),
        dtype=torch.float16).to(device)
    kernel = gaussian_gar * kernel
    list_kernel.append(kernel)
    del kernel, gaussian_gar
    
print()
print("Load model ...")
model = torch.load(args.dir_training + "/final_model.pth")
model = model.float()
model.to(device)
model.eval()

res = open(dir_res + '/res.csv', 'w')
fieldnames = ['Patient', 'Dice', 'clDice', 'Precision', 'Sensitivity', 'Euler_Number_Error', 'B0_error', 'B1_error', 'B2_error', 'Euler_Number_predicted',
              'Euler_Number_GT', 'B0_predicted', 'B0_GT', 'B1_predicted', 'B1_GT', 'B2_predicted', 'B2_GT', 'ASSD', 'HD', 'HD95', 'SurfaceDice']
writer = csv.DictWriter(res, fieldnames=fieldnames)
writer.writeheader()

list_dice = []
list_cldice = []
list_mcc = []
list_sens = []
list_spec = []
list_prec = []
list_euler_error = []
list_b0_error = []
list_b1_error = []
list_b2_error = []
list_euler_pred = []
list_euler_gt = []
list_b0_pred = []
list_b0_gt = []
list_b1_pred = []
list_b1_gt = []
list_b2_pred = []
list_b2_gt = []
list_assd = []
list_hd = []
list_hd95 = []
list_surface_dice = []
dict_summary = {}
with torch.no_grad():
    for batch, data in enumerate(test_data):
        
        string = data['filename'][0].split('.')
        name = string[0]
        
        print()
        print()
        print("Patient:" + name)
        
        X1 = data['image1']
        X2 = data['image2']
        X3 = data['image3']
        X4 = data['image4']
        
        y = data['GT']
        y = y.to(device)
        
        B, C, H, D, W = y.shape
        
        print()
        print("Original shape:")
        print(X1.shape)
        
        X1 = X1.float()
        X2 = X2.float()
        X3 = X3.float()
        X4 = X4.float()

        print()
        print("To GPU ...")
        X1 = X1.to(device)
        
        if args.augmentation:
            X2 = X2.to(device)
            X3 = X3.to(device)
            X4 = X4.to(device)
            
        print()
        print("Inference image1...")
        Y1, dmt = sliding_window_inference(inputs=X1, roi_size=(args.patch_size[0], args.patch_size[1], args.patch_size[2]), predictor=model, sw_batch_size=1, overlap=0.25, mode='gaussian', progress=True)
        if args.augmentation:
            print()
            print("Inference image2...")
            Y2, _ = sliding_window_inference(inputs=X2, roi_size=(args.patch_size[0], args.patch_size[1], args.patch_size[2]), predictor=model, sw_batch_size=1, overlap=0.25, mode='gaussian', progress=True)
        
            print()
            print("Inference image3...")
            Y3, _ = sliding_window_inference(inputs=X3, roi_size=(args.patch_size[0], args.patch_size[1], args.patch_size[2]), predictor=model, sw_batch_size=1, overlap=0.25, mode='gaussian', progress=True)
    
            print()
            print("Inference image4...")
            Y4, _ = sliding_window_inference(inputs=X4, roi_size=(args.patch_size[0], args.patch_size[1], args.patch_size[2]), predictor=model, sw_batch_size=1, overlap=0.25, mode='gaussian', progress=True)
           
            print()
            print("Flip back ...")
            Y2 = flip1(Y2[0]).view(1, 1, H, D, W)
            Y3 = flip2(Y3[0]).view(1, 1, H, D, W)
            Y4 = flip3(Y4[0]).view(1, 1, H, D, W)
            
        Y1 = sigmoid(Y1)
        if args.augmentation:
            Y2 = sigmoid(Y2)
            Y3 = sigmoid(Y3)
            Y4 = sigmoid(Y4)
        
            y_pred = torch.mean(torch.cat((Y1, Y2, Y3, Y4)), dim=0)
            
        else:
            y_pred = Y1[0]
              
        print()
        print("New shape:")
        print(y_pred.shape)
        
        #GEOMETRY-AWARE REFINEMENT
        print('GEOMETRY-AWARE REFINEMENT: applying soften ball...')
        ys, yv = torch.zeros_like(y_pred,dtype= torch.float16).to(device), torch.zeros_like(y_pred,dtype= torch.float16).to(device)
        dmt = torch.argmax(softmax(dmt), dim=1)
        print(torch.sum(dmt))
        print(torch.unique(dmt))
        skel = y_pred > 0.9
        skel = skel.type(torch.int).type(torch.float16)
        for radius in range(1,k):
             print(radius)
             kernel = list_kernel[radius-1]
             yv = dmt == radius
             print(torch.sum(yv))
             ys.add_(torch.clamp(torch.nn.functional.conv3d(skel*yv, kernel, padding=radius, groups=str_), 0, 1))
             del kernel

        ys = torch.clamp(ys, 0, 1)

        y_pred *= ys
        
        # Tresholding to binary segmentation
        y_pred = nn.functional.threshold(y_pred, threshold=0.5, value=0)
        y_pred = torch.where(y_pred > 0, torch.ones(y_pred.shape, dtype=torch.float, device=device), y_pred)
        
        # Transform to one-hot for Monai format
        y_pred = nn.functional.one_hot(y_pred[0].long())
        y_true = nn.functional.one_hot(y[0][0].long())
        
        # Permute axis and add channel to have [B, C, H, D, W]
        y_pred = y_pred.permute(3, 0, 1, 2).view(1, 2, H, D, W)
        y_true = y_true.permute(3, 0, 1, 2).view(1, 2, H, D, W)
        
        # Post-processing
        remove_small_objects_transform = RemoveSmallObjects(min_size=100)
        if args.postprocessing:
            print('Postprocessing...')
            y_true = remove_small_objects_transform(y_true)
            y_pred = remove_small_objects_transform(y_pred)
        
        # Monai metrics (on Pytorch Tensor)
        assd_metric = SurfaceDistanceMetric(symmetric=True)
        hd_metric = HausdorffDistanceMetric()
        hd95_metric = HausdorffDistanceMetric(percentile=95)
        surface_dice_metric = SurfaceDiceMetric(class_thresholds=[3])
        dice_metric = DiceMetric(include_background=False)
        
        # Compute metrics from monai
        assd = assd_metric(y_pred, y_true).item()
        hd = hd_metric(y_pred, y_true).item()
        hd95 = hd95_metric(y_pred, y_true).item()
        surface_dice = surface_dice_metric(y_pred, y_true).item()
        
        # From one-hot to standard format
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        
        # To numpy to compute numpy metrics
        y_pred = y_pred[0].detach().cpu().numpy()
        y_true = y_true[0].detach().cpu().numpy()
        
        print(y_pred.shape)
        print(y_true.shape)
        
        print("Compute metrics...")
        dice = dice_numpy(y_true, y_pred)
        # mcc = mcc_numpy(y_true, y_pred)
        cl_dice = cldice_numpy(y_true, y_pred)
        sens, spec, prec = sensitivity_specificity_precision(y_true, y_pred)
        euler_number_error, euler_number_gt, euler_number_pred = euler_number_error_numpy(y_true, y_pred, method='difference')
        b0_error, b0_gt, b0_pred = b0_error_numpy(y_true, y_pred, method='difference')
        b1_error, b1_gt, b1_pred = b1_error_numpy(y_true, y_pred, method='difference')
        b2_error, b2_gt, b2_pred = b2_error_numpy(y_true, y_pred, method='difference')

        list_dice.append(dice)
        list_cldice.append(cl_dice)
        # list_mcc.append(mcc)
        list_sens.append(sens)
        list_spec.append(spec)
        list_prec.append(prec)
        list_euler_error.append(euler_number_error)
        list_b0_error.append(b0_error)
        list_b1_error.append(b1_error)
        list_b2_error.append(b2_error)
        list_euler_pred.append(euler_number_pred)
        list_euler_gt.append(euler_number_gt)
        list_b0_pred.append(b0_pred)
        list_b0_gt.append(b0_gt)
        list_b1_pred.append(b1_pred)
        list_b1_gt.append(b1_gt)
        list_b2_pred.append(b2_pred)
        list_b2_gt.append(b2_gt)
        list_assd.append(assd)
        list_hd.append(hd)
        list_hd95.append(hd95)
        list_surface_dice.append(surface_dice)
        
        dict_csv = {
            "Patient": name,
            "Dice": dice,
            "clDice": cl_dice,
            "Precision": prec,
            "Sensitivity": sens,
            "Euler_Number_Error": euler_number_error,
            "B0_error": b0_error,
            "B1_error": b1_error,
            "B2_error": b2_error,
            "Euler_Number_predicted": euler_number_pred,
            "Euler_Number_GT": euler_number_gt,
            "B0_predicted": b0_pred,
            "B0_GT": b0_gt,
            "B1_predicted": b1_pred,
            "B1_GT": b1_gt,
            "B2_predicted": b1_pred,
            "B2_GT": b2_gt,
            "ASSD": assd,
            "HD": hd,
            "HD95": hd95,
            "SurfaceDice": surface_dice
        }
        writer.writerow(dict_csv)
        
        print()
        print("Dice")
        print(dice)
        print("clDice")
        print(cl_dice)
        print("Sens, Spec, Prec")
        print(sens, spec, prec)
        print("Euler_Number_Error")
        print(euler_number_error)
        print("B0_error")
        print(b0_error)
        print("B1_error")
        print(b1_error)
        print("B2_error")
        print(b2_error)
        print("Euler_Number_pred")
        print(euler_number_pred)
        print("Euler_Number_GT")
        print(euler_number_gt)
        print("B0_pred")
        print(b0_pred)
        print("B0_GT")
        print(b0_gt)
        print("B1_pred")
        print(b1_pred)
        print("B1_GT")
        print(b1_gt)
        print("B2_pred")
        print(b2_pred)
        print("B2_GT")
        print(b2_gt)
        print("ASSD")
        print(assd)
        print("HD")
        print(hd)
        print("HD95")
        print(hd95)
        print("SurfaceDice")
        print(surface_dice)
        
        image_path = os.path.join(dir_inputs, name + '.nii.gz')
        segmentation_path = os.path.join(dir_seg, name + '.nii.gz')
        image = nib.load(image_path)
        mra_headers = image.header
        affine = mra_headers.get_best_affine()
        
        img = nib.Nifti1Image(y_pred, affine=affine, header=mra_headers)
        nib.save(img, segmentation_path)
        
        
dict_csv = {
    "Patient": 'Mean',
    "Dice": np.mean(list_dice),
    "clDice": np.mean(list_cldice),
    "Precision": np.mean(list_prec),
    "Sensitivity": np.mean(list_sens),
    "Euler_Number_Error": np.mean(list_euler_error),
    "B0_error": np.mean(list_b0_error),
    "B1_error": np.mean(list_b1_error),
    "B2_error": np.mean(list_b2_error),
    "Euler_Number_predicted": np.mean(list_euler_pred),
    "Euler_Number_GT": np.mean(list_euler_gt),
    "B0_predicted": np.mean(list_b0_pred),
    "B0_GT": np.mean(list_b0_gt),
    "B1_predicted": np.mean(list_b1_pred),
    "B1_GT": np.mean(list_b1_gt),
    "B2_predicted": np.mean(list_b1_pred),
    "B2_GT": np.mean(list_b2_gt),
    "ASSD": np.mean(list_assd),
    "HD": np.mean(list_hd),
    "HD95": np.mean(list_hd95),
    "SurfaceDice": np.mean(list_surface_dice)
}
writer.writerow(dict_csv)

dict_csv = {
    "Patient": 'Std',
    "Dice": np.std(list_dice),
    "clDice": np.std(list_cldice),
    "Precision": np.std(list_prec),
    "Sensitivity": np.std(list_sens),
    "Euler_Number_Error": np.std(list_euler_error),
    "B0_error": np.std(list_b0_error),
    "B1_error": np.std(list_b1_error),
    "B2_error": np.std(list_b2_error),
    "Euler_Number_predicted": np.std(list_euler_pred),
    "Euler_Number_GT": np.std(list_euler_gt),
    "B0_predicted": np.std(list_b0_pred),
    "B0_GT": np.std(list_b0_gt),
    "B1_predicted": np.std(list_b1_pred),
    "B1_GT": np.std(list_b1_gt),
    "B2_predicted": np.std(list_b1_pred),
    "B2_GT": np.std(list_b2_gt),
    "ASSD": np.std(list_assd),
    "HD": np.std(list_hd),
    "HD95": np.std(list_hd95),
    "SurfaceDice": np.std(list_surface_dice)
}
writer.writerow(dict_csv)

res.close()

print()
print()
print("Mean Dice:")
print(np.mean(list_dice))
print("Mean clDice:")
print(np.mean(list_cldice))
print("Mean MCC:")
print(np.mean(list_mcc))
print("Mean Sens:")
print(np.mean(list_sens))
print("Mean Spec:")
print(np.mean(list_spec))
print("Mean Prec:")
print(np.mean(list_prec))
print("Mean Euler Number Error")
print(np.mean(list_euler_error))
print("Mean B0 Error")
print(np.mean(list_b0_error))
print("Mean B1 Error")
print(np.mean(list_b1_error))
print("Mean B2 Error")
print(np.mean(list_b2_error))
print("Mean Euler Number predicted")
print(np.mean(list_euler_pred))
print("Mean Euler Number GT")
print(np.mean(list_euler_gt))
print("Mean B0 pred")
print(np.mean(list_b0_pred))
print("Mean B0 GT")
print(np.mean(list_b0_gt))
print("Mean B1 pred")
print(np.mean(list_b1_pred))
print("Mean B1 GT")
print(np.mean(list_b1_gt))
print("Mean B2 pred")
print(np.mean(list_b2_pred))
print("Mean B2 GT")
print(np.mean(list_b2_gt))
print("ASSD")
print(np.mean(list_assd))
print("HD")
print(np.mean(list_hd))
print("HD95")
print(np.mean(list_hd95))
print("SurfaceDice")
print(np.mean(list_surface_dice))
