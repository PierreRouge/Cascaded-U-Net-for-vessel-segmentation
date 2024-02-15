#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:02:31 2022

@author: rouge
"""

import os
import numpy as np
import nibabel as nib
from skimage.morphology import skeletonize

dir_inputs = '/run/media/rouge/HDD_NVO/IXI_temporaire/IXI/IXI-MRA_annotations_Maria/GT/'
dir_outputs = '/run/media/rouge/HDD_NVO/IXI_temporaire/IXI/IXI-MRA_annotations_Maria/skeleton/'
if not os.path.exists(dir_outputs):
    os.makedirs(dir_outputs)
    
for (root, directory, file) in os.walk(dir_inputs):
    for f in file:
        GT_path = os.path.join(root, f)
        image = nib.load(GT_path)
        headers = image.header
        affine = headers.get_best_affine()
        image = np.array(image.get_fdata())
        skeleton = skeletonize(image) / 255
        skeleton_nii = nib.Nifti1Image(skeleton, affine=affine, header=headers)
        nib.save(skeleton_nii, dir_outputs + '/' + f[:-7] + '_skeleton.nii.gz')
        
        
        
        
