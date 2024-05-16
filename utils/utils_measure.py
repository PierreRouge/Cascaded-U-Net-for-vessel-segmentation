#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:41:34 2022

@author: rouge
"""
import os
import csv
import numpy as np
import nibabel as nib
import wandb

from skimage.measure import label, euler_number
from skimage.morphology import skeletonize, dilation, ball

# from medpy.metric.binary import hd, assd

def compute_haussdorf(dir_prediction, dir_GT, dir_outputs, term, wandbrun):
    list_hd = []

    file_hd = open(dir_outputs + "/res_hd.csv", "w")
    writer = csv.writer(file_hd)
    writer.writerow(["Patient", "Haussdorf distance"])
    file_hd.close()
    
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            string = f.split('.')
            name = string[0]
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, f[-20:-7] + '_' + term + '.nii.gz')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            GT = np.array(GT.get_fdata())
            prediction = np.array(prediction.get_fdata())
            haussdorf_distance = hd(prediction, GT, voxelspacing=(0.5134, 0.5134, 0.8000))
            
    
            
            file_hd = open(dir_outputs + "/res_hd.csv", "a")
            writer = csv.writer(file_hd)
            writer.writerow([f[:-7], haussdorf_distance])
            file_hd.close()
            
            list_hd.append(haussdorf_distance)
    list_hd = np.array(list_hd)
    hd_mean = np.mean(list_hd)
    hd_std = np.std(list_hd)
    incertitude = hd_std / np.sqrt(len(list_hd))
    
    file_hd = open(dir_outputs + "/res_hd.csv", "a")
    writer = csv.writer(file_hd)
    writer.writerow(["Mean", hd_mean])
    writer.writerow(["Std", hd_std])
    writer.writerow(["Incertitude", incertitude])
    file_hd.close()
    
    
    
    if wandbrun is not None:
        api = wandb.Api()
        run = api.run(wandbrun)
        run.summary["Mean hd Postprocessed"] = hd_mean
        run.summary.update()
    
    file_hd.close()
    
def compute_assd(dir_prediction, dir_GT, dir_outputs, term, wandbrun):
    list_assd = []
    
    file_assd = open(dir_outputs + "/res_assd.csv", "w")
    writer = csv.writer(file_assd)
    writer.writerow(["Patient", "ASSD"])
    file_assd.close()
    
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, f[-20:-7] + '_' + term + '.nii.gz')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            GT = np.array(GT.get_fdata())
            prediction = np.array(prediction.get_fdata())
            ass_distance = assd(prediction, GT, voxelspacing=(0.5134, 0.5134, 0.8000))
            
            file_assd = open(dir_outputs + "/res_assd.csv", "a")
            writer = csv.writer(file_assd)
            writer.writerow([f[:-7], ass_distance])
            file_assd.close()
            
            list_assd.append(ass_distance)
    list_assd = np.array(list_assd)
    assd_mean = np.mean(list_assd)
    assd_std = np.std(list_assd)
    incertitude = assd_std / np.sqrt(len(list_assd))
    
    file_assd = open(dir_outputs + "/res_assd.csv", "a")
    writer = csv.writer(file_assd)
    writer.writerow(["Mean", assd_mean])
    writer.writerow(["Std", assd_std])
    writer.writerow(["Incertitude", incertitude])
    file_assd.close()
    
    if wandbrun is not None:
        api = wandb.Api()
        run = api.run(wandbrun)
        run.summary["Mean assd Postprocessed"] = assd_mean
        run.summary.update()
    
    file_assd.close()
    
def dice_numpy(y_true, y_pred):
    """
    Compute dice on numpy array

    Parameters
    ----------
    y_true : Numpy array of size (dim1, dim2, dim3)
        Ground truth
    y_pred : Numpy array of size (dim1, dim2, dim3)
         Predicted segmentation

    Returns
    -------
    dice : Float
        Value of dice

    """
    epsilon = 1e-5
    numerator = 2 * (np.sum(y_true * y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    dice = (numerator + epsilon) / (denominator + epsilon)
    return dice

def overlap_numpy(y_true, y_pred):
    """
    Compute overlap dice on numpy array

    Parameters
    ----------
    y_true : Numpy array of size (dim1, dim2, dim3)
        Ground truth
    y_pred : Numpy array of size (dim1, dim2, dim3)
         Predicted segmentation

    Returns
    -------
    dice : Float
        Value of dice

    """
    epsilon = 1e-5
    structure = ball(radius=2)
    
    y_true = dilation(y_true, structure)
    y_pred = dilation(y_pred, structure)
    
    numerator = 2 * (np.sum(y_true * y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    dice = (numerator + epsilon) / (denominator + epsilon)
    return dice



def dice_numpy_skeleton(y_true, y_pred):
    epsilon = 1e-5
    y_true = skeletonize(y_true) / 255
    y_pred = skeletonize(y_pred) / 255
    numerator = 2 * (np.sum(y_true * y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    dice = (numerator + epsilon) / (denominator + epsilon)
    return dice


def tprec_numpy(y_true, y_pred):
    epsilon = 1e-5
    y_pred = skeletonize(y_pred) / 255
    numerator = np.sum(y_true * y_pred)
    denominator = np.sum(y_pred)
    tprec = (numerator + epsilon) / (denominator + epsilon)
    return tprec


def tsens_numpy(y_true, y_pred):
    epsilon = 1e-5
    y_true = skeletonize(y_true) / 255
    numerator = (np.sum(y_true * y_pred))
    denominator = np.sum(y_true)
    tsens = (numerator + epsilon) / (denominator + epsilon)
    return tsens


def cldice_numpy(y_true, y_pred):
    tprec = tprec_numpy(y_true, y_pred)
    tsens = tsens_numpy(y_true, y_pred)
    numerator = 2 * tprec * tsens
    denominator = tprec + tsens
    cldice = numerator / denominator
    return cldice


def sensitivity_specificity_precision(y_true, y_pred):
    tp = np.sum(y_pred * y_true)
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    sens = tp / np.sum(y_true)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    
    return sens, spec, prec


def mcc_numpy(y_true, y_pred):
    tp = np.sum(y_pred * y_true)
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    return mcc
    

def connex_component(image_nii):
    """
    Take a segmentation (nii format) and return the segmentation with each connex component labeled, the number of connex components and the biggest connex component.

    Parameters
    ----------
    image_nii : nii files
    input image (segmentation)

    Returns
    -------
    labeled_segmentation_nii : nii file
        Segmentation labeled (assign a nummber to each connex component).
    number_connex_component : int
        Number of different connex component
    principale_component_nii : nii file
        Biggest connex component

    """
    
    image_header = image_nii.header
    image_affine = image_header.get_best_affine()
    seg = image_nii.get_fdata()
    seg = np.array(seg, dtype=bool)
    labeled_segmentation, number_connex_component = label(seg, return_num=True)
    labeled_segmentation_nii = nib.Nifti1Image(labeled_segmentation, affine=image_affine, header=image_header)
    
    principale_component = np.copy(seg)
    composante_principale = 0
    max_composante_principale = 0
    histo = []
    for i in range(1, number_connex_component):
        card_comp = np.sum(labeled_segmentation == i)
        histo.append(card_comp)
        if card_comp > max_composante_principale:
            composante_principale = i
            max_composante_principale = card_comp
    histo.sort(reverse=True)
    
    principale_component[labeled_segmentation != composante_principale] = 0
    
    rate = np.sum(principale_component) / np.sum(seg)
    principale_component_nii = nib.Nifti1Image(principale_component, affine=image_affine, header=image_header)
    
    return labeled_segmentation_nii, number_connex_component, principale_component_nii, rate, histo


def metric_connex_component(y_true, y_pred, prec):
    
    # Compute labeled segmentation
    labeled_segmentation_pred, ncc_pred = label(y_pred, return_num=True)
    labeled_segmentation, ncc_gt = label(y_true, return_num=True)
    
    #Intersection between GT and segmentation
    inter = y_true * y_pred
    
    # Create a list containing the size of each connected component and an other list with the indices
    histo = []
    list_ind = []
    for i in range(1, ncc_pred):
        card_comp = np.sum(labeled_segmentation_pred == i)
        histo.append(card_comp)
        list_ind.append(i)
        
    histo = np.array(histo)
    list_ind = np.array(list_ind)
    
    # Sort the lists by the size of the component (from the biggest to the smallest)
    sort_indices = np.argsort(histo)
    histo = np.flip(histo[sort_indices])
    list_ind = np.flip(list_ind[sort_indices])
    
    # Compute RMCC_pred and RMCC_seg
    indices_ncc = np.zeros((448, 448, 128), dtype=bool)
    if ncc_gt < ncc_pred:
        for j in range(0, ncc_gt):
            indices_ncc = indices_ncc + (labeled_segmentation_pred == list_ind[j])
            
        rmcc_seg = np.sum(y_pred[indices_ncc]) / np.sum(y_pred)
        rmcc_gt = np.sum(inter[indices_ncc]) / np.sum(y_true)
        
    else:
        rmcc_seg = np.sum(y_pred) / np.sum(y_pred)
        rmcc_gt = np.sum(inter) / np.sum(y_true)
    
    #Compute number of connected component required to cover *rate* (percent) of the segmentation 
    i = 0
    rate = 0
    indices = np.zeros((448, 448, 128), dtype=bool)
    while i < ncc_pred - 1 and rate < prec:
        indices = indices + (labeled_segmentation_pred == list_ind[i])
        rate = np.sum(y_pred[indices]) / np.sum(y_pred)
        i += 1
     
    #Compute number of connected component required to cover *rate* (percent) of the segmentation 
    i2 = 0
    rate = 0
    indices = np.zeros((448, 448, 128), dtype=bool)
    while i2 < ncc_pred - 1 and rate < prec:
        indices = indices + (labeled_segmentation_pred == list_ind[i2])
        rate = np.sum(inter[indices]) / np.sum(y_true)
        i2 += 1
        
    return i, i2, ncc_pred, rmcc_seg, rmcc_gt


def euler_number_numpy(y):
    return euler_number(y)

def euler_number_numpy_v2(y):
    shape_ = y.shape
    new_shape = (shape_[0] * 2 + 1, shape_[1] * 2 + 1, shape_[2] * 2 + 1)
    CW = np.zeros(new_shape)
    for i in range(0, shape_[0]):
        for j in range(0, shape_[1]):
            for k in range(0, shape_[2]):
                a, b, c = 2*i + 1, 2*j + 1, 2*k + 1
                CW[a, b, c] = y[i, j, k]
                if CW[a, b, c] == 1:
                    for p in [-1, 0, 1]:
                        for q in [-1, 0, 1]:
                            for s in [-1, 0, 1]:
                                CW[a + p,b + q, c + s] = 1
              
    f0 = 0
    f1 = 0
    f2 = 0
    f3 = 0                             
    for i in range(0, shape_[0] * 2 + 1):
        for j in range(0, shape_[1] * 2 + 1):
            for k in range(0, shape_[2] * 2 +1):
                if (i % 2 == 0) and (j % 2 == 0)  and (k % 2 == 0):
                    f0 += CW[i, j, k]
                elif (i % 2 == 1) and (j % 2 == 1)  and (k % 2 == 1):
                    f3 += CW[i, j, k]
                elif ((i % 2 == 0) and (j % 2 == 1)  and (k % 2 == 1)) or ((i % 2 == 1) and (j % 2 == 0)  and (k % 2 == 1)) or ((i % 2 == 1) and (j % 2 == 1)  and (k % 2 == 0)):
                    f2 += CW[i, j, k]
                elif ((i % 2 == 1) and (j % 2 == 0)  and (k % 2 == 0)) or ((i % 2 == 0) and (j % 2 == 1)  and (k % 2 == 0)) or ((i % 2 == 0) and (j % 2 == 0)  and (k % 2 == 1)):   
                    f1 += CW[i, j, k]
    euler = f0 - f1 + f2 -f3
        
    return euler, f0, f1, f2, f3

def euler_number_error_numpy(y_true, y_pred, method='difference'):
    euler_number_true = euler_number(y_true)
    euler_number_pred = euler_number(y_pred)
    
    if method == 'difference' :
        euler_number_error = np.absolute(euler_number_true - euler_number_pred)
    
    elif method == 'relative' :
        euler_number_error = np.absolute(np.absolute(euler_number_true - euler_number_pred) / euler_number_true)
    
    return euler_number_error, euler_number_true, euler_number_pred
    

def b0_error_numpy(y_true, y_pred, method='difference'):
    _, ncc_true = label(y_true, return_num=True)
    _, ncc_pred = label(y_pred, return_num=True)
    
    b0_true= ncc_true
    b0_pred = ncc_pred
    
    if method == 'difference' :
        b0_error = np.absolute(b0_true - b0_pred)
    elif method == 'relative' :
       b0_error = np.absolute(b0_true - b0_pred) / b0_true
    
    return b0_error, b0_true, b0_pred

def b1_error_numpy(y_true, y_pred, method='difference'):
    
    euler_number_true = euler_number_numpy(y_true)
    euler_number_pred = euler_number_numpy(y_pred)
    
    _, ncc_true = label(y_true, return_num=True)
    _, ncc_pred = label(y_pred, return_num=True)
    
    b0_true= ncc_true
    b0_pred = ncc_pred
    
    y_true_inverse = np.ones(y_true.shape) - y_true
    y_pred_inverse = np.ones(y_pred.shape) - y_pred
    
    _, ncc_true = label(y_true_inverse, return_num=True)
    _, ncc_pred = label(y_pred_inverse, return_num=True)
    
    b2_true= ncc_true - 1
    b2_pred = ncc_pred - 1
    
    b1_true = b0_true + b2_true - euler_number_true
    b1_pred = b0_pred + b2_pred - euler_number_pred
    
    
    if method == 'difference' :
        b1_error = np.absolute(b1_true - b1_pred)
    elif method == 'relative' :
        b1_error = np.absolute(b1_true - b1_pred) / b1_true
    
    return b1_error, b1_true, b1_pred

def b2_error_numpy(y_true, y_pred, method='difference'):
    y_true_inverse = np.ones(y_true.shape) - y_true
    y_pred_inverse = np.ones(y_pred.shape) - y_pred
    
    _, ncc_true = label(y_true_inverse, return_num=True)
    _, ncc_pred = label(y_pred_inverse, return_num=True)
    
    b2_true= ncc_true - 1
    b2_pred = ncc_pred - 1
    
    if method == 'difference' :
        b2_error = np.absolute(b2_true - b2_pred)
        
    elif method == 'relative' :
         b2_error = np.absolute(b2_true - b2_pred) / b2_true
         
    return b2_error, b2_true, b2_pred
        

  
def info_connex_component(dir_inputs, dir_outputs, wandbrun):
    """

    Parameters
    ----------
    dir_inputs : str
        Path to the directory contening the segmentation
    dir_outputs : str
        Path to the directory where we want the results

    Returns
    -------
    None.

    """
    
    if not os.path.exists(dir_outputs):
        os.makedirs(dir_outputs)
    
    dir_connex = dir_outputs + '/connex_components'
    dir_princ = dir_outputs + '/principale_component'
    if not os.path.exists(dir_connex):
        os.makedirs(dir_connex)
    if not os.path.exists(dir_princ):
        os.makedirs(dir_princ)
    
    file_pc = open(dir_outputs + "/res_pc.csv", "w")
    writer = csv.writer(file_pc)
    writer.writerow(["Patient", "Connex Components"])
    file_pc.close()
    
    file_rate = open(dir_outputs + "/res_rate.csv", "w")
    writer = csv.writer(file_rate)
    writer.writerow(["Patient", "Rate principale component"])
    file_rate.close()
    
    list_connex = []
    list_rate = []
    for (root, directory, file) in os.walk(dir_inputs):
        for f in file:
            segmentation_path = os.path.join(root, f)
            image = nib.load(segmentation_path)
            labeled_segmentation_nii, number_connex_component, principale_component_nii, rate, histo = connex_component(image)
            
            list_connex.append(number_connex_component)
            list_rate.append(rate)
            
            nib.save(labeled_segmentation_nii, dir_connex + '/' + f[:-7] + '_labeled')
            nib.save(principale_component_nii, dir_princ + '/' + f[:-7] + '_principale_component')
            
            file_pc = open(dir_outputs + "/res_pc.csv", "a")
            writer = csv.writer(file_pc)
            writer.writerow([f[:-7], number_connex_component])
            file_pc.close()
            
            file_rate = open(dir_outputs + "/res_rate.csv", "a")
            writer = csv.writer(file_rate)
            writer.writerow([f[:-7], rate])
            file_rate.close()
            
            
    list_connex = np.array(list_connex)
    list_rate = np.array(list_rate)
    
    mean_connex = np.mean(list_connex)
    std = np.std(list_connex)
    incertitude = std / np.sqrt(len(list_connex))
       
    file_pc = open(dir_outputs + "/res_pc.csv", "a")
    writer = csv.writer(file_pc)
    writer.writerow(['Mean', mean_connex])
    writer.writerow(['Std', std])
    writer.writerow(['Incertitude', incertitude])
    file_pc.close()
    
    mean_rate = np.mean(list_rate)
    std = np.std(list_rate)
    incertitude = std / np.sqrt(len(list_rate))
    
    file_rate = open(dir_outputs + "/res_rate.csv", "a")
    writer = csv.writer(file_rate)
    writer.writerow(['Mean', mean_rate])
    writer.writerow(['Std', std])
    writer.writerow(['Incertitude', incertitude])
    file_rate.close()
    
    if wandbrun is not None:
        api = wandb.Api()
        run = api.run(wandbrun)
        run.summary["Mean number of connex components"] = mean_connex
        run.summary["Mean rate principale component"] = mean_rate
        run.summary.update()
     
# Compute true positive and false positive            
def get_map_tp_fp(prediction, ground_truth):
    """
    Take a predicted segmentation and the associated ground truth and compute true positive and false positive
    
    Parameters
    ----------
    prediction : numpy array
        Predicted segmentation
    ground_truth : numpy array
        Ground truth segmentation

    Returns
    -------
    map_ : numpy array
        Array contening 0 in the background, 1 for true positive and 2 for false positive

    """
    image_shape = prediction.shape
    map_ = np.zeros(image_shape)
    map_[prediction * ground_truth == 1] = 1
    map_[np.logical_and(prediction == 1, ground_truth == 0)] = 0
    
    return map_


# Compute true positive and false negative
def get_map_tp_fn(prediction, ground_truth):
    """
    Take a predicted segmentation and the associated ground truth and compute true positive and false negative

    Parameters
    ----------
    prediction : numpy array
        Predicted segmentation
    ground_truth : numpy array
        Ground truth segmentation

    Returns
    -------
    map_ : numpy array
        Array contening 0 in the background, 1 for true positive and 2 for false negative

    """
    image_shape = prediction.shape
    map_ = np.zeros(image_shape)
    map_[prediction * ground_truth == 1] = 1
    map_[np.logical_and(prediction == 0, ground_truth == 1)] = 2
    
    return map_


# Compute true positive, false positive and false negative
def get_map_tp_fp_fn(prediction, ground_truth):
    """
    Take a predicted segmentation and the associated ground truth and compute true positive, false positive and false negative

    Parameters
    ----------
    prediction : numpy array
        Predicted segmentation
    ground_truth : numpy array
        Ground truth segmentation

    Returns
    -------
    map_tp : numpy array
        Array contening 0 in the background, 1 for true positive
    map_fp : numpy array
        Array contening 0 in the background, 1 for false positive
    map_fn : numpy array
        Array contening 0 in the background, 1 for false negative

    """
    image_shape = prediction.shape
    map_tp = np.zeros(image_shape)
    map_fp = np.zeros(image_shape)
    map_fn = np.zeros(image_shape)
    map_tp[prediction * ground_truth == 1] = 1
    map_fp[np.logical_and(prediction == 1, ground_truth == 0)] = 2
    map_fn[np.logical_and(prediction == 0, ground_truth == 1)] = 3
    
    return map_tp, map_fp, map_fn

   
def built_map_tp_fp_fn(dir_prediction, dir_GT, dir_outputs, term):
    
    if not os.path.exists(dir_outputs):
        os.makedirs(dir_outputs)
    
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, f[-20:-7] + '.nii.gz')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            headers = prediction.header
            affine = headers.get_best_affine()
            prediction = prediction.get_fdata()
            GT = GT.get_fdata()
            prediction = np.array(prediction, dtype=bool)
            GT = np.array(GT, dtype=int)
            
            map_tp, map_fp, map_fn = get_map_tp_fp_fn(prediction, GT)
            map_tp_nii = nib.Nifti1Image(map_tp, affine=affine, header=headers)
            map_fp_nii = nib.Nifti1Image(map_fp, affine=affine, header=headers)
            map_fn_nii = nib.Nifti1Image(map_fn, affine=affine, header=headers)
            nib.save(map_tp_nii, dir_outputs + '/' + f[:-7] + '_map_tp.nii')
            nib.save(map_fp_nii, dir_outputs + '/' + f[:-7] + '_map_fp.nii')
            nib.save(map_fn_nii, dir_outputs + '/' + f[:-7] + '_map_fn.nii')


def compute_dice(dir_prediction, dir_GT, dir_outputs, term, wandbrun):
    list_dice = []
    
    file_dice = open(dir_outputs + "/res_dice.csv", "w")
    writer = csv.writer(file_dice)
    writer.writerow(["Patient", "Dice"])
    file_dice.close()
    
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            string = f.split('.')
            name = string[0]
            
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, name + term + '.nii.gz')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            GT = np.array(GT.get_fdata())
            prediction = np.array(prediction.get_fdata())
            dice = dice_numpy(GT, prediction)
            
            file_dice = open(dir_outputs + "/res_dice.csv", "a")
            writer = csv.writer(file_dice)
            writer.writerow([name, dice])
            file_dice.close()
            
            list_dice.append(dice)
    list_dice = np.array(list_dice)
    dice_mean = np.mean(list_dice)
    dice_std = np.std(list_dice)
    incertitude = dice_std / np.sqrt(len(list_dice))
    
    file_dice = open(dir_outputs + "/res_dice.csv", "a")
    writer = csv.writer(file_dice)
    writer.writerow(["Mean", dice_mean])
    writer.writerow(["Std", dice_std])
    writer.writerow(["Incertitude", incertitude])
    file_dice.close()
    
    if wandbrun is not None:
        api = wandb.Api()
        run = api.run(wandbrun)
        run.summary["Dice"] = dice_mean
        run.summary.update()
    
    file_dice.close()


def compute_cldice(dir_prediction, dir_GT, dir_outputs, term, wandbrun):
    list_cldice = []
    
    file_cldice = open(dir_outputs + "/res_cldice.csv", "w")
    writer = csv.writer(file_cldice)
    writer.writerow(["Patient", "clDice"])
    file_cldice.close()
    
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            string = f.split('.')
            name = string[0]
            
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, name + '_' + term + '.nii.gz')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            GT = np.array(GT.get_fdata())
            prediction = np.array(prediction.get_fdata())
            cldice = cldice_numpy(GT, prediction)
            
            file_cldice = open(dir_outputs + "/res_cldice.csv", "a")
            writer = csv.writer(file_cldice)
            writer.writerow([name, cldice])
            file_cldice.close()
    
            list_cldice.append(cldice)
            
    list_cldice = np.array(list_cldice)
    cldice_mean = np.mean(list_cldice)
    cldice_std = np.std(list_cldice)
    incertitude = cldice_std / np.sqrt(len(list_cldice))
    
    file_cldice = open(dir_outputs + "/res_cldice.csv", "a")
    writer = csv.writer(file_cldice)
    writer.writerow(["Mean", cldice_mean])
    writer.writerow(["Std", cldice_std])
    writer.writerow(["Incertitude", incertitude])
    file_cldice.close()
    
    if wandbrun is not None:
        api = wandb.Api()
        run = api.run(wandbrun)
        run.summary["clDice"] = cldice_mean
        run.summary.update()
    
    file_cldice.close()    
    
def compute_mcc(dir_prediction, dir_GT, dir_outputs, term, wandbrun):
    list_mcc = []
    
    file_mcc = open(dir_outputs + "/res_mcc.csv", "w")
    writer = csv.writer(file_mcc)
    writer.writerow(["Patient", "MCC"])
    file_mcc.close()
    
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            string = f.split('.')
            name = string[0]
            
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, name + term + '.nii.gz')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            GT = np.array(GT.get_fdata(), dtype=np.dtype(np.float32))
            prediction = np.array(prediction.get_fdata(), np.dtype(np.float32))
            mcc = mcc_numpy(GT, prediction)
            
            file_mcc = open(dir_outputs + "/res_mcc.csv", "a")
            writer = csv.writer(file_mcc)
            writer.writerow([name, mcc])
            file_mcc.close()
            
            list_mcc.append(mcc)
    list_mcc = np.array(list_mcc)
    mcc_mean = np.mean(list_mcc)
    mcc_std = np.std(list_mcc)
    incertitude = mcc_std / np.sqrt(len(list_mcc))
    
    file_mcc = open(dir_outputs + "/res_mcc.csv", "a")
    writer = csv.writer(file_mcc)
    writer.writerow(["Mean", mcc_mean])
    writer.writerow(["Std", mcc_std])
    writer.writerow(["Incertitude", incertitude])
    file_mcc.close()
    
    if wandbrun is not None:
        api = wandb.Api()
        run = api.run(wandbrun)
        run.summary["MCC"] = mcc_mean
        run.summary.update()
    
    file_mcc.close()


def compute_dice_skeleton(dir_prediction, dir_GT, dir_outputs):
    list_dice = []
    file_dice = open(dir_outputs + "/res_dice_skeleton.txt", "w")
    file_dice.write('Dice sur squelettes volumes de tests :\n')
    file_dice.close()
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, f[-20:-7] + '_GT.nii')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            GT = np.array(GT.get_fdata())
            prediction = np.array(prediction.get_fdata())
            dice = dice_numpy_skeleton(GT, prediction)
            file_dice = open(dir_outputs + "/res_dice_skeleton.txt", "a")
            file_dice.write(f[:-7] + ':' + str(dice) + '\n')
            file_dice.close()
            list_dice.append(dice)
            list_dice.append(dice)
    list_dice = np.array(list_dice)
    dice_mean = np.mean(list_dice)
    dice_std = np.std(list_dice)
    incertitude = dice_std / np.sqrt(len(list_dice))
    file_dice = open(dir_outputs + "/res_dice_skeleton.txt", "a")
    file_dice.write('Dice mean : ' + str(dice_mean) + '\n')
    file_dice.write('Dice std : ' + str(dice_std) + '\n')
    file_dice.write('Incertitude : ' + str(incertitude) + '\n')
    file_dice.close()


def compute_tprec(dir_prediction, dir_GT, dir_outputs, wandbrun):
    
    list_tprec = []
    
    file_tprec = open(dir_outputs + "/res_tprec.csv", "w")
    writer = csv.writer(file_tprec)
    writer.writerow(["Patient", "tprec"])
    file_tprec.close()
    
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, f[-20:-7] + '_GT.nii.gz')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            GT = np.array(GT.get_fdata())
            prediction = np.array(prediction.get_fdata())
            tprec = tprec_numpy(GT, prediction)
            
            file_tprec = open(dir_outputs + "/res_tprec.csv", "a")
            writer = csv.writer(file_tprec)
            writer.writerow([f[:-7], tprec])
            file_tprec.close()
            
            list_tprec.append(tprec)
            list_tprec.append(tprec)
    list_tprec = np.array(list_tprec)
    tprec_mean = np.mean(list_tprec)
    tprec_std = np.std(list_tprec)
    incertitude = tprec_std / np.sqrt(len(list_tprec))

    file_tprec = open(dir_outputs + "/res_tprec.csv", "a")
    writer = csv.writer(file_tprec)
    writer.writerow(["Mean", tprec_mean])
    writer.writerow(["Std", tprec_std])
    writer.writerow(["Incertitude", incertitude])
    file_tprec.close()
    
    if wandbrun is not None:
        api = wandb.Api()
        run = api.run(wandbrun)
        run.summary["Tprec Mean"] = tprec_mean
        run.summary.update()
    

def compute_tsens(dir_prediction, dir_GT, dir_outputs, wandbrun):
    list_tsens = []
    
    file_tsens = open(dir_outputs + "/res_tsens.csv", "w")
    writer = csv.writer(file_tsens)
    writer.writerow(["Patient", "tsens"])
    file_tsens.close()
    
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, f[-20:-7] + '_GT.nii.gz')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            GT = np.array(GT.get_fdata())
            prediction = np.array(prediction.get_fdata())
            tsens = tsens_numpy(GT, prediction)
        
            file_tsens = open(dir_outputs + "/res_tsens.csv", "a")
            writer = csv.writer(file_tsens)
            writer.writerow([f[:-7], tsens])
            file_tsens.close()
            
            list_tsens.append(tsens)
            list_tsens.append(tsens)
            
    list_tsens = np.array(list_tsens)
    tsens_mean = np.mean(list_tsens)
    tsens_std = np.std(list_tsens)
    incertitude = tsens_std / np.sqrt(len(list_tsens))
    
    file_tsens = open(dir_outputs + "/res_tsens.csv", "a")
    writer = csv.writer(file_tsens)
    writer.writerow(["Mean", tsens_mean])
    writer.writerow(["Std", tsens_std])
    writer.writerow(["Incertitude", incertitude])
    file_tsens.close()
    
    if wandbrun is not None:
        api = wandb.Api()
        run = api.run(wandbrun)
        run.summary["Tsens Mean"] = tsens_mean
        run.summary.update()





def compute_sens_spec_prec(dir_prediction, dir_GT, dir_outputs, term, wandbrun):
    list_sens = []
    list_spec = []
    list_prec = []
    
    file_sens = open(dir_outputs + "/res_sens.csv", "w")
    writer = csv.writer(file_sens)
    writer.writerow(["Patient", "sens"])
    file_sens.close()
    
    file_spec = open(dir_outputs + "/res_spec.csv", "w")
    writer = csv.writer(file_spec)
    writer.writerow(["Patient", "spec"])
    file_spec.close()
    
    file_prec = open(dir_outputs + "/res_prec.csv", "w")
    writer = csv.writer(file_prec)
    writer.writerow(["Patient", "prec"])
    file_prec.close()
    
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            string = f.split('.')
            name = string[0]
            
            
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, name + '_' + term + '.nii.gz')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            GT = np.array(GT.get_fdata(), dtype=bool)
            prediction = np.array(prediction.get_fdata(), dtype=bool)
            sens, spec, prec = sensitivity_specificity_precision(GT, prediction)
            
            file_sens = open(dir_outputs + "/res_sens.csv", "a")
            writer = csv.writer(file_sens)
            writer.writerow([name, sens])
            file_sens.close()
            
            file_spec = open(dir_outputs + "/res_spec.csv", "a")
            writer = csv.writer(file_spec)
            writer.writerow([name, spec])
            file_spec.close()
        
            file_prec = open(dir_outputs + "/res_prec.csv", "a")
            writer = csv.writer(file_prec)
            writer.writerow([name, prec])
            file_prec.close()
            
            file_prec.close()
            list_sens.append(sens)
            list_spec.append(spec)
            list_prec.append(prec)
    
    list_sens = np.array(list_sens)
    sens_mean = np.mean(list_sens)
    sens_std = np.std(list_sens)
    sens_incertitude = sens_std / np.sqrt(len(list_sens))
    
    file_sens = open(dir_outputs + "/res_sens.csv", "a")
    writer = csv.writer(file_sens)
    writer.writerow(["Mean", sens_mean])
    writer.writerow(["Std", sens_std])
    writer.writerow(["Incertitude", sens_incertitude])
    file_sens.close()

    
    list_spec = np.array(list_spec)
    spec_mean = np.mean(list_spec)
    spec_std = np.std(list_spec)
    spec_incertitude = spec_std / np.sqrt(len(list_spec))
    
    file_spec = open(dir_outputs + "/res_spec.csv", "a")
    writer = csv.writer(file_spec)
    writer.writerow(["Mean", spec_mean])
    writer.writerow(["Std", spec_std])
    writer.writerow(["Incertitude", spec_incertitude])
    file_spec.close()
 
    list_prec = np.array(list_prec)
    prec_mean = np.mean(list_prec)
    prec_std = np.std(list_prec)
    prec_incertitude = prec_std / np.sqrt(len(list_prec))
    
    file_prec = open(dir_outputs + "/res_prec.csv", "a")
    writer = csv.writer(file_prec)
    writer.writerow(["Mean", prec_mean])
    writer.writerow(["Std", prec_std])
    writer.writerow(["Incertitude", prec_incertitude])
    file_prec.close()
    
    if wandbrun is not None:
        api = wandb.Api()
        run = api.run(wandbrun)
        run.summary["Sensitivity"] = sens_mean
        run.summary["Specifiticity"] = spec_mean
        run.summary["Precision"] = prec_mean
        run.summary.update()
  
    
def dice_principale_component(y_true, y_pred):
    y_true_label, y_true_num = label(y_true, return_num=True)
    y_pred_label, y_pred_num = label(y_pred, return_num=True)
    
    composante_principale = 0
    max_composante_principale = 0
    for i in range(1, y_true_num):
        card_comp = np.sum(y_true_label == i)
        if card_comp > max_composante_principale:
            composante_principale = i
            max_composante_principale = card_comp
    
    y_true_label[y_true_label != composante_principale] = 0
    y_true_label[y_true_label == composante_principale] = 1
    
    composante_principale = 0
    max_composante_principale = 0
    for i in range(1, y_pred_num):
        card_comp = np.sum(y_pred_label == i)
        if card_comp > max_composante_principale:
            composante_principale = i
            max_composante_principale = card_comp
    
    y_pred_label[y_pred_label != composante_principale] = 0
    y_pred_label[y_pred_label == composante_principale] = 1
    
    dice = dice_numpy(y_true_label, y_pred_label)
    
    return dice
   
    
def compute_dice_principale_component(dir_prediction, dir_GT, dir_outputs):
    list_dice = []
    file_dice = open(dir_outputs + "/res_dice_pc.txt", "w")
    file_dice.write('Dice sur composante principale des volumes de tests :\n')
    file_dice.close()
    for (root, directory, file) in os.walk(dir_prediction):
        for f in file:
            path_prediction = os.path.join(dir_prediction, f)
            path_GT = os.path.join(dir_GT, f[13:-7] + '_GT.nii')
            prediction = nib.load(path_prediction)
            GT = nib.load(path_GT)
            GT = np.array(GT.get_fdata())
            prediction = np.array(prediction.get_fdata(), dtype=bool)
            dice = dice_principale_component(GT, prediction)
            file_dice = open(dir_outputs + "/res_dice_pc.txt", "a")
            file_dice.write(f[:-7] + ':' + str(dice) + '\n')
            file_dice.close()
            list_dice.append(dice)
    list_dice = np.array(list_dice)
    dice_mean = np.mean(list_dice)
    dice_std = np.std(list_dice)
    incertitude = dice_std / np.sqrt(len(list_dice))
    file_dice = open(dir_outputs + "/res_dice_pc.txt", "a")
    file_dice.write('Dice mean : ' + str(dice_mean) + '\n')
    file_dice.write('Dice std : ' + str(dice_std) + '\n')
    file_dice.write('Incertitude : ' + str(incertitude) + '\n')
    file_dice.close() 


    