import sys;
import os;
import tensorflow as tf;
slim = tf.contrib.slim;
import numpy as np;
from distutils.util import strtobool;
import glob;
from time import sleep

from ..sa_net_arch_utilities import CNNArchUtils;
#from sa_net_loss_func import CNNLossFuncHelper, CostFuncTypes;
from ..sa_net_arch import AbstractCNNArch;
from ..sa_net_cost_func import AbstractCostFunc;

import sys;
#sys.path.append("/pylon5/ac3uump/shahira/tf-slim_models/models/research/slim")
#from nets.inception_resnet_v2 import inception_resnet_v2
#from nets.inception_resnet_v2 import inception_resnet_v2_arg_scope
#from nets.inception_resnet_v2 import inception_resnet_v2_base
#from sa_networks.inception_resnet_v2 import inception_resnet_v2
#from sa_networks.inception_resnet_v2 import inception_resnet_v2_arg_scope
#from sa_networks.inception_resnet_v2 import inception_resnet_v2_base
#from sa_networks.inception_v3 import inception_v3
#from sa_networks.inception_v3 import inception_v3_arg_scope
from ..sa_networks.vgg import vgg_16
from ..sa_networks.vgg import vgg_arg_scope

class VGG16ClassifierArch(AbstractCNNArch):
    def __init__(self, n_channels, n_classes, model_out_path, model_base_filename, model_restore_filename, cost_func:AbstractCostFunc, kwargs):
        args = {'input_img_width':-1, 'input_img_height':-1, 'pretrained':False, 'freeze_layers':-1, 'extra_end_layer':-1, 
            'get_features': 'False', 'official_checkpoint':'False', 'l2_reg':0.0, 'version':'0', 'prefix':None};
        #print(kwargs)
        args.update(kwargs);
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.model_out_path = model_out_path;
        self.model_base_filename = model_base_filename;
        self.model_restore_filename = model_restore_filename;
        self.cost_func = cost_func;
        self.current_model_checkpoint_path = None;
        self.input_img_width = int(args['input_img_width']);
        self.input_img_height = int(args['input_img_height']);
        self.official_checkpoint = bool(strtobool(args['official_checkpoint']));
        self.L2_reg = float(args['l2_reg']);
        self.version = str(args['version']); # 0, add_prev_feat_w1x1
        self.prefix = args['prefix']; # 0, add_prev_feat_w1x1
        #print(self.L2_reg)
        #print(args['l2_reg'])
        self.input_x = tf.placeholder(tf.float32, shape=(None, self.input_img_height, self.input_img_width, n_channels));
        self.labels = tf.placeholder(tf.float32, shape=(None, n_classes));
        self.isTest = tf.placeholder(tf.bool);
        #self.isTraining = tf.placeholder(tf.bool);
        self.isTraining = tf.math.logical_not(self.isTest);
        self.dropout = tf.placeholder(tf.float32);
        #self.class_weights = tf.Variable(tf.ones([n_classes]));
        self.epochs_count = tf.Variable(0);

        self.logits, self.end_points = self.create_model(self.input_x, self.isTraining, kwargs);
        if('vgg_16/fc7' in self.end_points ):
            self.features = self.end_points['vgg_16/fc7'];
        elif(not(self.prefix is None) and self.prefix+'/vgg_16/fc7' in self.end_points ):
            self.features = self.end_points[self.prefix+'/vgg_16/fc7'];
        else:
            self.features = None;
        if('AuxLogits' in self.end_points):
            self.aux_logits = self.end_points['AuxLogits'];
        else:
            self.aux_logits = None;
        self.cost = self.cost_func.calc_cost(self.logits, self.labels);
        if(self.aux_logits is not None):
            self.cost += self.cost_func.calc_cost(self.aux_logits, self.labels);
        if(self.L2_reg > 0):
            self.cost += tf.add_n(slim.losses.get_regularization_losses());  # L2 regularization loss
        #self.prediction_softmax = self.get_prediction_softmax(self.logits);
        #self.prediction_class = self.get_class_prediction(self.logits);
        self.correct_pred = self.get_correct_prediction(self.logits, self.labels);
        self.accuracy = self.get_accuracy();

        restore_exclude_list = ['Variable', 'Variable_1', 'Variable_2', 'Variable_3', 'vgg_16/fc8'];
        if(not(self.prefix is None)):
            restore_exclude_list.append(self.prefix+'/fc8');
            if(self.official_checkpoint):
                restore_exclude_list.append(self.prefix+'/*');
        if(self.version == 'add_prev_feat_w1x1'):
            restore_exclude_list.append('vgg_16/Conv2d_pre_1x1');
            restore_exclude_list.append('vgg_16/Conv2d_pre_1x1/biases');
            restore_exclude_list.append('vgg_16/fc6');
            restore_exclude_list.append('vgg_16/fc7');
        variables_to_restore = slim.get_variables_to_restore(exclude=restore_exclude_list);
        #print ([v.name for v in variables_to_restore]);
        #self.saver = tf.train.Saver(var_list =variables_to_restore[1:], max_to_keep=100000);
        self.saver_official = tf.train.Saver(var_list =variables_to_restore, max_to_keep=100000);
        variables_to_restore = slim.get_variables_to_restore(exclude=['Variable', 'Variable_1', 'Variable_2', 'Variable_3']);
        self.saver = tf.train.Saver(var_list =variables_to_restore, max_to_keep=100000);

    def create_model(self, input_x, isTraining, kwargs):
        # predefined list of arguments
        args = { 'dropout':0.75, 'dropout_inner':1.0};

        
        args.update(kwargs);

        # read extra argument
        dropout = float(args['dropout']);
        dropout_inner = float(args['dropout_inner']);
        keep_prob = dropout;


        #with slim.arg_scope(vgg_arg_scope(weight_decay=self.L2_reg)):
        #    logits, end_points = vgg_16(input_x, num_classes=self.n_classes, is_training=isTraining, dropout_keep_prob=keep_prob, dropout_keep_prob_inner=dropout_inner, weight_decay=self.L2_reg, version=self.version)

        if(self.prefix is None or self.official_checkpoint):
            print('in vgg')
            with slim.arg_scope(vgg_arg_scope(weight_decay=self.L2_reg)):
                logits, end_points = vgg_16(input_x, num_classes=self.n_classes, is_training=isTraining, dropout_keep_prob=keep_prob, dropout_keep_prob_inner=dropout_inner, weight_decay=self.L2_reg, version=self.version)

        if(self.prefix):
            print('in prefix')
            with tf.variable_scope(self.prefix):
                with slim.arg_scope(vgg_arg_scope(weight_decay=self.L2_reg)):
                    logits, end_points = vgg_16(input_x, num_classes=self.n_classes, is_training=isTraining, dropout_keep_prob=keep_prob, dropout_keep_prob_inner=dropout_inner, weight_decay=self.L2_reg, version=self.version)

    
                
        return logits, end_points;

    def restore_model(self, sess):
        print('before restore');
        if(self.model_restore_filename is None):
            self.filepath = None;
            ##debug
            print('self.model_restore_filename is None')
            return None;
        self.filepath = self.model_restore_filename;
        #if(os.path.isfile(self.model_restore_filename)):
        #    self.filepath = self.model_restore_filename;
        #else:
        #    self.filepath = os.path.join(self.model_out_path, self.model_restore_filename + '.ckpt');
        ###debug
        ##print('filepath =', self.filepath )
        #if(not os.path.isfile(self.filepath)):
        #    filepath_pattern = os.path.join(self.model_out_path, self.model_base_filename + '*.ckpt');
        #    list_of_files = glob.glob(filepath_pattern);
        #    if(len(list_of_files) <= 0):
        #        return None;
        #    self.filepath = max(list_of_files);
        #    print(self.filepath);
        #    if(not os.path.isfile(self.filepath)):
        #        return None;

        if(self.official_checkpoint):
            self.saver_official.restore(sess, self.filepath);
            self.official_checkpoint = False;
        else:
            print('restore filepath = ', self.filepath );
            self.saver.restore(sess, self.filepath);
        print('after restore');

    def save_model(self, sess, optimizer, epoch, suffix=""):
        postfix = '_epoch_{:04d}'.format(epoch) ;
        if(suffix is not None):
            postfix += suffix;
        self.filepath = os.path.join(self.model_out_path, self.model_base_filename+ postfix + '.ckpt');
        print('self.filepath = ', self.filepath);
        self.saver.save(sess, self.filepath);
        return self.filepath;

    def rename_model_variables(self, sess):
        restore_exclude_list = ['Variable', 'Variable_1', 'Variable_2', 'Variable_3', 'vgg_16/fc8', self.prefix+'/*']
        vars = slim.get_variables_to_restore(exclude=restore_exclude_list);
        exclude_list = [];
        for v in vars:
            print(v.name);    
            var = tf.get_default_graph().get_tensor_by_name(v.name);            
            var2 = tf.get_default_graph().get_tensor_by_name(self.prefix+"/"+v.name);            
            print(var.name);    
            print(var2.name);    
            print(var.eval());    
            print(var2.eval());    
            tf.assign(var2, var).eval(); 

            var3 = tf.get_default_graph().get_tensor_by_name(self.prefix+"/"+v.name);  
          
            print(var3.name);    
            print(var3.eval());    
            

        save_exclude_list = ['Variable', 'Variable_1', 'Variable_2', 'Variable_3', 'vgg_16/*']
        variables_to_restore = slim.get_variables_to_restore(exclude=save_exclude_list);
        saver = tf.train.Saver(var_list =variables_to_restore, max_to_keep=100000);
        new_model_filepath = os.path.join(self.model_out_path, self.model_base_filename + '.ckpt');
        saver.save(sess, new_model_filepath);
        return ;

    def get_trainable_var_list(self, sess=None):
        var_list = None;
        #if(not (var_include_name_pattern is None)):
        #    tvars = tf.trainable_variables()
        #    var_list = [var for var in tvars if var_include_name_pattern in var.name];
        return var_list;

            
