#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:10:23 2023

@author: bbnc
"""

import os

# Example for training
command_line = 'python Main_train.py --gpu_id 0 --smpl_name Microtubules --save_suffix _0 --net_type unet --max_iter 100000 --preload_data_flag'
command_line = 'python Main_train_3D.py --gpu_id 0 --smpl_name Lyso-3D --save_suffix _0 --net_type unet --max_iter 100000 --preload_data_flag'

os.system(command_line)


# Example for denoising
command_line = 'python Main_test.py --gpu_id 0 --smpl_name Microtubules --model_name Microtubules --overlap_ratio 0.2'
command_line = 'python Main_test_3D.py --gpu_id 0 --smpl_name Lyso-3D  --model_name Lyso-3D --overlap_ratio 0.2'
os.system(command_line)

