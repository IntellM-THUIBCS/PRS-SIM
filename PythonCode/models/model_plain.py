from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.nn.parallel import DataParallel  # , DistributedDataParallel

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import SparseLoss, VggLoss, CharbonnierLoss, SSIMLoss, CharbonnierEdgeLoss, PerceptualLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils.utils_general import crop_center


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.netG = define_G(opt).to(self.device)
        # self.netG = DataParallel(self.netG)

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.opt_train = self.opt['train']    # training option
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    # ----------------------------------------
    # save model
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type in ['l1', 'L1', 'charbonnierloss', 'CharbonnierLoss']:
            self.G_lossfn = CharbonnierLoss().to(self.device)
        elif G_lossfn_type in ['mse', 'l2', 'L2', 'MSE']:
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type[0:6] == 'MseVgg':
            self.G_lossfn = VggLoss(VGG_loss_weight=float(G_lossfn_type[6:])).to(self.device)
        elif G_lossfn_type[0:9] == 'MseSparse':  # type MseSparse1.0_0.3 | type MseSparse0.5_0.1
            self.G_lossfn = SparseLoss(order=float(G_lossfn_type[9:12]),
                                       sparse_loss_weight=float(G_lossfn_type[13:])).to(self.device)
        elif G_lossfn_type[:19] == 'charbonnierlossedge':
            self.G_lossfn = CharbonnierEdgeLoss(weight=float(G_lossfn_type[19:])).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))
    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_HL=True, need_R = False, test_patch_size=-1, test_patch_range=[], test_z_range=[]):
        if test_patch_size < 0:
            if test_patch_range:
                if test_z_range:
                    if need_HL:
                        self.L = data['L'][...,test_z_range[0]:test_z_range[1],test_patch_range[0]:test_patch_range[1],
                                       test_patch_range[2]:test_patch_range[3]].to(self.device)
                        self.H = data['H'][...,test_z_range[0]:test_z_range[1],test_patch_range[0]:test_patch_range[1],
                                       test_patch_range[2]:test_patch_range[3]].to(self.device)
                    if need_R:
                        self.R = data['R'][...,test_z_range[0]:test_z_range[1],test_patch_range[0]:test_patch_range[1],
                                       test_patch_range[2]:test_patch_range[3]].to(self.device)    
        
                else:
                    if need_HL:
                        self.L = data['L'][...,test_patch_range[0]:test_patch_range[1],
                                       test_patch_range[2]:test_patch_range[3]].to(self.device)
                        self.H = data['H'][...,test_patch_range[0]:test_patch_range[1],
                                       test_patch_range[2]:test_patch_range[3]].to(self.device)
                    if need_R:
                        self.R = data['R'][...,test_patch_range[0]:test_patch_range[1],
                                       test_patch_range[2]:test_patch_range[3]].to(self.device)
            else:                          
                if need_HL:
                    self.L = data['L'].to(self.device)  
                    self.H = data['H'].to(self.device)
                if need_R:
                    self.R = data['R'].to(self.device)
            
        elif test_patch_range:
            if need_HL:
                self.L = crop_center(data['L'],test_patch_size)[...,test_patch_range[0]:test_patch_range[1],
                                   test_patch_range[2]:test_patch_range[3]].to(self.device)
                self.H = crop_center(data['H'],test_patch_size)[...,test_patch_range[0]:test_patch_range[1],
                               test_patch_range[2]:test_patch_range[3]].to(self.device)
            if need_R:
                self.R = crop_center(data['R'],test_patch_size)[...,test_patch_range[0]:test_patch_range[1],
                               test_patch_range[2]:test_patch_range[3]].to(self.device)
        else:
            if need_HL:
                self.L = crop_center(data['L'],test_patch_size).to(self.device)
                self.H = crop_center(data['H'],test_patch_size).to(self.device)
            if need_R:
                self.R = crop_center(data['R'],test_patch_size).to(self.device)
            
            

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.E = self.netG(self.L)
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        G_loss.backward()

        # ------------------------------------
        # clip_graddata['L']
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()
 
    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self, H_input=False, R_input = False):
        assert (not (H_input and R_input)), "Error: only one input can be designed"
        self.netG.eval()
        with torch.no_grad():
            if (not R_input) and (not H_input):
                self.E = self.netG(self.L)
            elif H_input:
                self.E = self.netG(self.H)
            else:
                self.E = self.netG(self.R)
        self.netG.train()



    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def test_split(self, minsize=192):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=2, min_size=minsize, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_HL=True, need_R=False):
        out_dict = OrderedDict()
        
        out_dict['E'] = self.E.detach()[0].float().cpu()        
        if need_HL:
            out_dict['L'] = self.L.detach()[0].float().cpu()
            out_dict['H'] = self.H.detach()[0].float().cpu()

        if need_R:
            out_dict['R'] = self.R.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_HL=True, need_R=False):
        out_dict = OrderedDict()
        
        out_dict['E'] = self.E.detach().float().cpu()
        if need_HL:
            out_dict['L'] = self.L.detach().float().cpu()
            out_dict['H'] = self.H.detach().float().cpu()
        if need_R:
            out_dict['R'] = self.R.detach().float().cpu()
            
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
