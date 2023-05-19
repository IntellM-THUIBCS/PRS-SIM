import os.path
import math
import argparse
import time
import random
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import tifffile

import scipy.io as sio


'''
# ----------------------------------------
# Step--1 (prepare opt)
# ----------------------------------------
'''
print('\nprepare opt\n')

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=str, default='1', help='the id of GPU')

parser.add_argument('--smpl_dir', type=str, default='./SIM_data', help='the folder saveing the sample data')
parser.add_argument('--smpl_name', type=str, default='Microtubules', help='the name of the sample')
parser.add_argument('--net_type', type=str, default='unet', help='network type')
parser.add_argument('--save_suffix', type=str, default='_0', help='the suffix of the saving model')

parser.add_argument('--test_patch_size', type=int, default=128, help='the crop size of the image used for validation')
parser.add_argument('--max_iter', type=int, default=100000, help='the maximum iteration number for training')

parser.add_argument('--preload_data_flag', action='store_true', help='the flag indicating whether to preload all training datas in the memory')

params = parser.parse_args()
# debug
params.preload_data_flag = True

opt_name = 'train_'+ params.net_type +'.json'

opt = option.parse('options'+'/'+opt_name, is_train=True, 
                   smpl_name=params.smpl_name, net_type=params.net_type, save_suffix=params.save_suffix)

opt['datasets']['train']['dataroot'] = os.path.join(params.smpl_dir,params.smpl_name)
opt['datasets']['test']['dataroot'] = os.path.join(params.smpl_dir,params.smpl_name)
opt['datasets']['train']['preload_data_flag'] = params.preload_data_flag
opt['datasets']['test']['preload_data_flag'] = False

opt['datasets']['test']['test_cell_count'] = 1

torch.cuda.set_device(int(params.gpu_id))    
util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

# ----------------------------------------
# update opt
# ----------------------------------------


# ----------------------------------------
# save opt to  a '../option.json' file
# ----------------------------------------
option.save(opt)

# ----------------------------------------
# return None for missing key
# ----------------------------------------
opt = option.dict_to_nonedict(opt)
if opt['sleep_time'] >= 1:
    print('sleep {:.2f} hours'.format(opt['sleep_time']/3600))
    time.sleep(opt['sleep_time'])

# ----------------------------------------
# configure logger
# ----------------------------------------
logger_name = 'train'
utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
logger = logging.getLogger(logger_name)
logger.info(option.dict2str(opt))
writer = SummaryWriter(os.path.join(opt['path']['log']))

# ----------------------------------------
# seed
# ----------------------------------------
seed = opt['train']['manual_seed']
if seed is None:
    seed = random.randint(1, 10000)
logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

'''
# ----------------------------------------
# Step--2 (creat dataloader)
# ----------------------------------------
'''
print('\ncreat dataloader\n')

# ----------------------------------------
# 1) create_dataset
# 2) creat_dataloader for train and test
# ----------------------------------------
dataset_type = opt['datasets']['train']['dataset_type']
for phase, dataset_opt in opt['datasets'].items():
    if phase == 'train':
        train_set = define_Dataset(dataset_opt)
        train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
        train_loader = DataLoader(train_set,
                                  batch_size=dataset_opt['dataloader_batch_size'],
                                  shuffle=dataset_opt['dataloader_shuffle'],
                                  num_workers=dataset_opt['dataloader_num_workers'],
                                  drop_last=True,  # use or abandon the last minibatch
                                  pin_memory=True) # using swamp memory
    elif phase == 'test':
        test_set = define_Dataset(dataset_opt)
        test_loader = DataLoader(test_set, batch_size=1,
                                 shuffle=False, num_workers=1,
                                 drop_last=False, pin_memory=True)
    else:
        raise NotImplementedError("Phase [%s] is not recognized." % phase)

# len(train_set)
# len(test_set)
# tmp1 = train_set.__getitem__(0)
# tmp2 = test_set.__getitem__(0)

'''G_regularizer_orthstep
# ----------------------------------------
# Step--3 (initialize model)
# ----------------------------------------
'''
print('\ninitialize model\n')

model = define_Model(opt)

logger.info(model.info_network())
model.init_train()
logger.info(model.info_params())

'''
# ----------------------------------------
# Step--4 (main training)
# ----------------------------------------
'''
print('\nmain training\n')

current_step = 0
for epoch in range(opt['epoch_num']):  # keep running
    for i, train_data in enumerate(train_loader):

        current_step += 1

        # -------------------------------
        # 1) update learning rate
        # -------------------------------
        model.update_learning_rate(current_step)

        # -------------------------------
        # 2) feed patch pairs
        # -------------------------------
        model.feed_data(train_data)

        # -------------------------------
        # 3) optimize parameters
        # -------------------------------
        model.optimize_parameters(current_step)

        # -------------------------------
        # 4) training information
        # -------------------------------
        if current_step % opt['train']['checkpoint_print'] == 0:
            logs = model.current_log()  # such as loss
            message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
            for k, v in logs.items():  # merge log information into message
                message += '{:s}: {:.3e} '.format(k, v)
                writer.add_scalar('{:s}'.format(k), v, current_step)
            logger.info(message)
            writer.add_scalar('train_loss', model.log_dict['G_loss'], current_step)
            writer.add_scalar('lr', model.current_learning_rate(), current_step)

        # -------------------------------
        # 5) save model
        # -------------------------------
        if current_step % opt['train']['checkpoint_save'] == 0:
            logger.info('Saving the model.')
            model.save(current_step)
                
        # -------------------------------
        # 6) validation
        # -------------------------------
        if current_step % opt['train']['checkpoint_test'] == 0:
            
            idx = 0
            for test_data in test_loader:
                idx += 1
                
                img_name = test_data['L_path']['sim'][0].split('/')[-2]

                img_dir = os.path.join(opt['path']['images'], img_name)
                util.mkdir(img_dir)

                model.feed_data(test_data,test_patch_size=params.test_patch_size,need_R=True)
                model.test(R_input=True)

                visuals = model.current_visuals(need_R=True)
                
                E_img = util.tensor2float(visuals['E'])
                R_img = util.tensor2float(visuals['R'])
                
                E_img[E_img<0] = 0
                R_img[R_img<0] = 0

                # -----------------------
                # save estimated image E
                # -----------------------
                save_img_path_tif = os.path.join(img_dir, 'E_{:s}_{:d}.tif'.format(img_name, current_step))
                tifffile.imsave(save_img_path_tif, np.uint16(E_img))
                
                if current_step == opt['train']['checkpoint_test']:                                      
                    save_img_path_tif = os.path.join(img_dir, 'R_{:s}_{:d}.tif'.format(img_name, current_step))
                    tifffile.imsave(save_img_path_tif,np.uint16(R_img))

     
        if current_step == params.max_iter:
            exit()
        


