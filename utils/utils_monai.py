#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:11:02 2022

@author: rouge
"""
import torch

from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from typing import Mapping, Hashable, Dict

import sys
sys.path.append('..')
from utils.utils_pytorch import skeletonize_numpy, skeletonize_tensor


class skeletonized(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, datatype='torch'):
        super().__init__(keys, allow_missing_keys)
        self.datatype = datatype
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if self.datatype == 'torch':
                d[key] = skeletonize_tensor(d[key])
            elif self.datatype == 'numpy':
                d[key] = skeletonize_numpy(d[key])
            else:
                raise Exception("Sorry wrong datatype for skeletonization")
        return d
