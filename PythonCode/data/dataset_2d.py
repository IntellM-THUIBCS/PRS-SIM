import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import torch
import glob
import os

class DatasetPlain(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetPlain, self).__init__()
        print('Get L/H for image-to-image mapping.')
        self.opt = opt
        self.n_channels_in = opt['n_channels_in'] if opt['n_channels_in'] else 1
        self.n_channels_out = opt['n_channels_out'] if opt['n_channels_out'] else 1
        assert self.n_channels_in == self.n_channels_out
        self.n_channels = self.n_channels_in
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 64
        
        self.phase = self.opt['phase']
        self.preload_data_flag = self.opt['preload_data_flag']
        self.dataroot = self.opt['dataroot']
        
        self.train_cell_count = self.opt['train_cell_count'] if self.opt['train_cell_count'] else -1
        self.test_cell_count = self.opt['test_cell_count'] if self.opt['test_cell_count'] else -1
        
        self.view_count = 4
        self.edge_margin = 10
        # ------------------------------------
        # get the path
        # ------------------------------------
        
        if self.opt['phase'] == 'train':
            self.paths_V = [[] for _ in range(self.view_count)] 
        
        if self.opt['phase'] == 'test':
            self.paths_R = []  
        
        sub_dirs = sorted(glob.glob(self.dataroot+'/*'))
        
        if self.phase == 'train':
             for view_id in range(self.view_count):
                 tmp_paths = [{'sim':os.path.join(sub_dir,'view'+str(view_id+1)+'.tif')} for sub_dir in sub_dirs]
                 if self.train_cell_count > 0:
                     tmp_paths = tmp_paths[:self.train_cell_count]
                 self.paths_V[view_id].extend(tmp_paths[:])
                 
  
        if self.phase == 'test':
            tmp_paths = [{'sim':os.path.join(sub_dir,'raw'+'.tif')} for sub_dir in sub_dirs]
            if self.test_cell_count > 0:
                tmp_paths = tmp_paths[:self.test_cell_count]
            self.paths_R.extend(tmp_paths)
                
        # ------------------------------------
        # check the path
        # ------------------------------------
        if self.opt['phase'] == 'train':
            for view_id in range(self.view_count):            
                assert self.paths_V[view_id], 'Error: V path is needed but it is empty.'
                assert len(self.paths_V[view_id])==len(self.paths_V[0]), 'Error: the file number of each view must be the same.'
                
        if self.opt['phase'] == 'test':
            assert self.paths_R, 'Error: R path is needed but it is empty.'
        
               
        if self.phase =='train' and self.preload_data_flag:
            self.imgs_V = [[] for _ in range(self.view_count)] 
            
            for view_id in range(self.view_count):
                print('Loading data of view %d, please wait ...'%(view_id+1))

                for path_V in self.paths_V[view_id]:
                    img_sim = util.read_img(path_V['sim'])
                    self.imgs_V[view_id].append(img_sim)


    def __getitem__(self, index):

        if self.opt['phase'] == 'train':
            
            view_mode = np.random.randint(0, 8)
            
            file_id = index // self.view_count            
            
            # ------------------------------------
            # get H L image
            # ------------------------------------
            
            if view_mode == 0:
                H_path = self.paths_V[0][file_id]
                L_path = self.paths_V[1][file_id]
                
                if self.preload_data_flag:
                    img_H = self.imgs_V[0][file_id]
                    img_L = self.imgs_V[1][file_id]
                else:                    
                    img_H = util.read_img(H_path['sim'])
                    img_L = util.read_img(L_path['sim'])
                                    
            elif view_mode == 1:
                H_path = self.paths_V[1][file_id]
                L_path = self.paths_V[2][file_id]
                if self.preload_data_flag:
                    img_H = self.imgs_V[1][file_id]
                    img_L = self.imgs_V[2][file_id]
                else:                    
                    img_H = util.read_img(H_path['sim'])
                    img_L = util.read_img(L_path['sim'])
                    
            elif view_mode == 2:
                H_path = self.paths_V[2][file_id]
                L_path = self.paths_V[3][file_id]
                if self.preload_data_flag:
                    img_H = self.imgs_V[2][file_id]
                    img_L = self.imgs_V[3][file_id]
                else:                    
                    img_H = util.read_img(H_path['sim'])
                    img_L = util.read_img(L_path['sim'])
                    
            elif view_mode == 3:
                H_path = self.paths_V[3][file_id]
                L_path = self.paths_V[0][file_id]
                if self.preload_data_flag:
                    img_H = self.imgs_V[3][file_id]
                    img_L = self.imgs_V[0][file_id]
                else:
                    img_H = util.read_img(H_path['sim'])
                    img_L = util.read_img(L_path['sim'])

            elif view_mode == 4:
                H_path = self.paths_V[1][file_id]
                L_path = self.paths_V[0][file_id]
                if self.preload_data_flag:
                    img_H = self.imgs_V[1][file_id]
                    img_L = self.imgs_V[0][file_id]
                else:                    
                    img_H = util.read_img(H_path['sim'])
                    img_L = util.read_img(L_path['sim']) 
                    
            elif view_mode == 5:
                H_path = self.paths_V[2][file_id]
                L_path = self.paths_V[1][file_id]
                if self.preload_data_flag:
                    img_H = self.imgs_V[2][file_id]
                    img_L = self.imgs_V[1][file_id]
                else:                    
                    img_H = util.read_img(H_path['sim'])
                    img_L = util.read_img(L_path['sim'])
                    
            elif view_mode == 6:
                H_path = self.paths_V[3][file_id]
                L_path = self.paths_V[2][file_id]
                if self.preload_data_flag:
                    img_H = self.imgs_V[3][file_id]
                    img_L = self.imgs_V[2][file_id]
                else:                    
                    img_H = util.read_img(H_path['sim'])
                    img_L = util.read_img(L_path['sim'])
                    
            elif view_mode == 7:
                H_path = self.paths_V[0][file_id]
                L_path = self.paths_V[3][file_id]
                if self.preload_data_flag:
                    img_H = self.imgs_V[0][file_id]
                    img_L = self.imgs_V[3][file_id]
                else:
                    img_H = util.read_img(H_path['sim'])
                    img_L = util.read_img(L_path['sim'])
            else:
                raise NotImplementedError            

            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0+self.edge_margin, max(0, H - self.patch_size - self.edge_margin))
            rnd_w = random.randint(0+self.edge_margin, max(0, W - self.patch_size - self.edge_margin))
            
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.single2tensor3(patch_L), util.single2tensor3(patch_H)
            
            img_R = torch.zeros(img_L.shape,dtype=img_L.dtype)
            R_path = L_path
            
        if self.opt['phase'] == 'test':
            # ------------------------------------
            # get R image
            # ------------------------------------
            R_path = self.paths_R[index]
            
            img_R = util.read_img(R_path['sim'])

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_R = util.single2tensor3(img_R)
            
            img_L = torch.zeros(img_R.shape,dtype=img_R.dtype)
            img_H = torch.zeros(img_R.shape,dtype=img_R.dtype)
            L_path = R_path
            H_path = R_path

        return {'L': img_L, 'H': img_H, 'R': img_R, 'L_path': L_path, 'H_path': H_path, 'R_path': R_path}

    def __len__(self):
        if self.opt['phase'] == 'train':
            return 4*len(self.paths_V[0])
        else:
            return len(self.paths_R)
