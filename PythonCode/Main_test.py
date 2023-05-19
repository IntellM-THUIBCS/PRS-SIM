#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:10:23 2023

@author: bbnc
"""

import os.path
import math
import argparse
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model


import matplotlib.pyplot as plt

import tifffile

'''
# ----------------------------------------
# Step--1 (prepare opt)
# ----------------------------------------
'''

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=str, default='1', help='the id of GPU')

parser.add_argument('--smpl_dir', type=str, default='./SIM_data', help='the folder saveing the sample data')
# parser.add_argument('--smpl_name', type=str, default='Microtubules_test', help='the name of the sample')
parser.add_argument('--smpl_name', type=str, default='CCPs_test', help='the name of the sample')
parser.add_argument('--net_type', type=str, default='unet', help='network type') 

# parser.add_argument('--model_dir', type=str, default='./pretrained_models/Microtubules_unet_0/models', help='the folder saveing the model file')
parser.add_argument('--model_dir', type=str, default='./pretrained_models/CCPs_unet_0/models', help='the folder saveing the model file')
parser.add_argument('--model_name', type=str, default='100000_G', help='the name of the model file (.pth)')

parser.add_argument('--test_patch_size', type=int, default=1004, help='the crop size of the image')
parser.add_argument('--model_patch_size', type=int, default=128, help='the patch size of each tile')

parser.add_argument('--overlap_ratio', type=float, default=0.2, help='the overlap ratio between adjacent tiles')

params = parser.parse_args()

opt_name = 'train_'+ params.net_type +'.json'
opt = option.parse('options'+'/'+opt_name, is_train=False, 
                   smpl_name=params.smpl_name, net_type=params.net_type)

opt['datasets']['test']['dataroot'] = os.path.join(params.smpl_dir,params.smpl_name)
opt['datasets']['test']['preload_data_flag'] = False

torch.cuda.set_device(int(params.gpu_id))  
opt = option.dict_to_nonedict(opt)

'''
# # ----------------------------------------
# # Step--2 (load model)
# # ----------------------------------------
# '''

model_path = os.path.join(params.model_dir,params.model_name +'.pth')

assert os.path.exists(model_path), 'The designated model does not exist!'
opt['path']['pretrained_netG'] = model_path

opt = option.dict_to_nonedict(opt)

'''
# ----------------------------------------
# Step--3 (initialize seed)
# ----------------------------------------
'''

seed = opt['train']['manual_seed']
if seed is None:
    seed = random.randint(1, 10000)
# logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

'''
# ----------------------------------------
# Step--4 (load dataset)
# ----------------------------------------
'''
dataset_opt = opt['datasets']['test']
test_set = define_Dataset(dataset_opt)
test_loader = DataLoader(test_set, batch_size=1,
                         shuffle=False, num_workers=1,
                         drop_last=False, pin_memory=True)

print('The total image number to be denoised is '+ str(len(test_set)))
'''
# ----------------------------------------
# Step--5 (initialize model)
# ----------------------------------------
'''
model = define_Model(opt)
model.init_train()

'''
# ----------------------------------------
# Step--6 (perform the test)
# ----------------------------------------
'''
save_dir = os.path.join('./denoised_results','smpl_' + params.smpl_name  + '_' + 'model_' + params.model_name)
util.mkdir(save_dir)

idx = 0
for test_data in test_loader:    
    idx += 1
    
    print('Processing ' + '%02d'%(idx) + ' out of ' + '%02d'%(len(test_loader))+' !')
    img_name = test_data['L_path']['sim'][0].split('/')[-2]
    
    img_dir = os.path.join(save_dir, img_name)
    util.mkdir(img_dir)
    
    # test each patch and then stitch them together
    E_img = np.zeros((1,params.test_patch_size,params.test_patch_size), dtype=np.float32)
    R_img = np.zeros((1,params.test_patch_size,params.test_patch_size), dtype=np.float32)
    
    rr_list = list(range(0,params.test_patch_size-params.model_patch_size+1,int(params.model_patch_size/2)))
    if rr_list[-1] != params.test_patch_size-params.model_patch_size:
        rr_list.append(params.test_patch_size-params.model_patch_size)
    
    cc_list = list(range(0,params.test_patch_size-params.model_patch_size+1,int(params.model_patch_size/2)))
    if cc_list[-1] != params.test_patch_size-params.model_patch_size:
        cc_list.append(params.test_patch_size-params.model_patch_size)  

    for rr in rr_list:
        for cc in cc_list:
            if rr == 0:
                rr_min = 0
                rr_min_patch = 0
            else:
                rr_min = rr + int(params.model_patch_size*params.overlap_ratio)
                rr_min_patch = int(params.model_patch_size*params.overlap_ratio)
            
            if rr + params.model_patch_size == params.test_patch_size:
                rr_max = params.test_patch_size
                rr_max_patch = params.model_patch_size
            else:
                rr_max = rr + params.model_patch_size - int(params.model_patch_size*params.overlap_ratio)
                rr_max_patch = params.model_patch_size - int(params.model_patch_size*params.overlap_ratio)       
            
            if cc == 0:
                cc_min = 0
                cc_min_patch = 0
            else:
                cc_min = cc + int(params.model_patch_size*params.overlap_ratio)
                cc_min_patch = int(params.model_patch_size*params.overlap_ratio)
            
            if cc + params.model_patch_size == params.test_patch_size:
                cc_max = params.test_patch_size
                cc_max_patch = params.model_patch_size
            else:
                cc_max = cc + params.model_patch_size - int(params.model_patch_size*params.overlap_ratio)
                cc_max_patch = params.model_patch_size - int(params.model_patch_size*params.overlap_ratio)
            
            test_patch_range = [rr,rr+params.model_patch_size,cc,cc+params.model_patch_size]
            
            model.feed_data(test_data,test_patch_size=-1,need_R=True,test_patch_range=test_patch_range)
            model.test(R_input=True)
            
            visuals = model.current_visuals(need_R=True)
    
            E_img_p = util.tensor2float(visuals['E'])
            R_img_p = util.tensor2float(visuals['R'])
            
            E_img[0,rr_min:rr_max,cc_min:cc_max] = E_img_p[0,rr_min_patch:rr_max_patch,cc_min_patch:cc_max_patch]
            R_img[0,rr_min:rr_max,cc_min:cc_max] = R_img_p[0,rr_min_patch:rr_max_patch,cc_min_patch:cc_max_patch]
        
        
    # save imgs   
    E_img[E_img<0] = 0
    R_img[R_img<0] = 0
    
    save_img_path_tif = os.path.join(img_dir, 'E.tif')
    tifffile.imsave(save_img_path_tif, np.uint16(E_img))
       
    save_img_path_tif = os.path.join(img_dir, 'R.tif')
    tifffile.imsave(save_img_path_tif,np.uint16(R_img))
