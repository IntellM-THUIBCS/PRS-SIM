#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:14:09 2021

@author: bbnc
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def crop_center(img, crop_size):
    h = img.shape[-2]
    w = img.shape[-1]

    return img[...,h//2-crop_size//2:h//2+crop_size//2,w//2-crop_size//2:w//2+crop_size//2]