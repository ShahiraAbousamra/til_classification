#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.cm as CM

import numpy as np
import time
import torch
import torch.nn as nn
import os
#import visdom
import random
#from tqdm import tqdm as tqdm
import sys;
import math
#import skimage.io as io
#from scipy import ndimage
#from scipy.misc import imresize
import configparser;

from .resnet34_arch import TILClassifier
#from my_dataset_til import TILDataset
from .model_til_resnet34_external_input import TILClassifierExternalInputModel

def load_model(config_filepath):

    # read the config file
    config = configparser.ConfigParser();
    config.read(config_filepath);  

    if('model_restore_filename' in config['DEFAULT']):
        model_restore_filename = config['DEFAULT']['model_restore_filename'].strip();
    else:
        model_restore_filename = None;

    # Tester config
    #tester_params = dict(config.items('TESTER'));
    tester_config = config['TESTER'];
    is_binary_output = tester_config.getboolean('is_binary_output');
    threshold = float(tester_config['threshold'].strip());

    model_param_path = model_restore_filename;      

    gt_multiplier = 1    
    gpu_or_cpu='cuda' # use cuda or cpu
    lr                = 0.00005 #e0
    #lr                = 0.000005 #e4
    #lr                = 0.00000005 
    #lr                = 0.00005
    #lr                = 0.0005
    #lr                = 0.001
    batch_size        = 1
    momentum          = 0.95
    epochs            = 200
    steps             = [-1,1,100,150]
    scales            = [1,1,1,1]
    workers           = 4
    seed              = time.time()
    print_freq        = 30 

    dropout_keep_prob = 1.0
    #initial_pad = 94
    initial_pad = 126
    #initial_pad = 46
    #initial_pad = 58
    interpolate = 'False'
    conv_init = 'he'
    n_classes = 1
    n_channels = 1
    #lmda=[1,1,1,10];
    #threshold = 0.5
    threshold = 0
    
    lamda_topo            = 0; # e0
    #lamda_topo            = 0.5; # e4
    lamda_topo            = 1; # e4
    lamda_dice            = 1;
    sub_patch_border_width = 5
    topo_size         = 100; 
    mm=1

    start_epoch = 0

    #vis=visdom.Visdom()
    device=torch.device(gpu_or_cpu)
    torch.cuda.manual_seed(seed)
    model=TILClassifier(kwargs={'dropout_keep_prob':dropout_keep_prob, 'initial_pad':initial_pad, 'interpolate':interpolate, 'conv_init':conv_init, 'n_classes':n_classes, 'n_channels':n_channels})
    if(not (model_param_path is None)):
        model.load_state_dict(torch.load(model_param_path), strict=True);
        print('model loaded')
        ## init scale 1 with scale 2
        #model.output_layer_1.load_state_dict(model.output_layer_2.state_dict())
        #model.scale_1.load_state_dict(model.scale_2.state_dict())
        #print('model loaded2')
        #torch.save(model.state_dict(),os.path.join(checkpoints_save_path, 'epoch_'+str(0)+".pth")) # save only if get better error
    model.to(device)
    model.eval()
    model_runner = TILClassifierExternalInputModel(model, device, is_binary_output=is_binary_output, threshold=threshold, max_side=100, resize_side=100)
    return model_runner


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    