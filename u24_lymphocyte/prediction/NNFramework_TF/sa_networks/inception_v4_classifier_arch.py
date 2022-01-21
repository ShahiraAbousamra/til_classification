# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import sys;
import os;
import tensorflow as tf;
slim = tf.contrib.slim;
import numpy as np;
from distutils.util import strtobool;
import glob;

from ..sa_net_arch_utilities import CNNArchUtils;
#from sa_net_loss_func import CNNLossFuncHelper, CostFuncTypes;
from ..sa_net_arch import AbstractCNNArch;
from ..sa_net_cost_func import AbstractCostFunc;

import sys;
#sys.path.append("/pylon5/ac3uump/shahira/tf-slim_models/models/research/slim")
#from nets.inception_resnet_v2 import inception_resnet_v2
#from nets.inception_resnet_v2 import inception_resnet_v2_arg_scope
#from nets.inception_resnet_v2 import inception_resnet_v2_base
#from sa_networks.inception_resnet_v2 import inception_v2
#from sa_networks.inception_resnet_v2 import inception_resnet_v2_arg_scope
#from sa_networks.inception_resnet_v2 import inception_resnet_v2_base
from .inception_v4 import inception_v4
from .inception_v4 import inception_v4_arg_scope
#from sa_networks.resnet_v1 import resnet_v1_50
#from sa_networks.resnet_v1 import resnet_arg_scope

class InceptionV4ClassifierArch(AbstractCNNArch):
    def __init__(self, n_channels, n_classes, model_out_path, model_base_filename, model_restore_filename, cost_func:AbstractCostFunc, kwargs):
        args = {'input_img_width':-1, 'input_img_height':-1, 'pretrained':False, 'freeze_layers':-1, 'extra_end_layer':-1, 
            'get_features': 'False', 'official_checkpoint':'False'};

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
        self.input_x = tf.placeholder(tf.float32, shape=(None, self.input_img_height, self.input_img_width, n_channels));
        self.labels = tf.placeholder(tf.float32, shape=(None, n_classes));
        self.isTest = tf.placeholder(tf.bool);
        #self.isTraining = tf.placeholder(tf.bool);
        self.isTraining = tf.math.logical_not(self.isTest);
        self.dropout = tf.placeholder(tf.float32);
        #self.class_weights = tf.Variable(tf.ones([n_classes]));
        self.epochs_count = tf.Variable(0);

        self.logits, self.end_points = self.create_model(self.input_x, self.isTraining, kwargs);
        if('AuxLogits' in self.end_points):
            self.aux_logits = self.end_points['AuxLogits'];
        else:
            self.aux_logits = None;
        self.cost = self.cost_func.calc_cost(self.logits, self.labels);
        if(self.aux_logits is not None):
            self.cost += self.cost_func.calc_cost(self.aux_logits, self.labels);
        #self.prediction_softmax = self.get_prediction_softmax(self.logits);
        #self.prediction_class = self.get_class_prediction(self.logits);
        self.correct_pred = self.get_correct_prediction(self.logits, self.labels);
        self.accuracy = self.get_accuracy();

        variables_to_restore = slim.get_variables_to_restore(exclude=['Variable', 'Variable_1', 'Variable_2', 'Variable_3', \
            'InceptionV4/AuxLogits', 'InceptionV4/Logits'
            #, 'InceptionV4/Conv2d_1a_3x3/biases', 'InceptionV4/Conv2d_2a_3x3/biases', 'InceptionV4/Conv2d_2b_3x3/biases', 'InceptionV4/Mixed_3a/Branch_1/Conv2d_0a_3x3/biases'
                ])
        #print(type(variables_to_restore));
        variables_to_restore2 = [];
        for v in variables_to_restore:
            #print(v.name);
            if('biases' not in v.name):
                variables_to_restore2.append(v);
        #for v in variables_to_restore2:
        #    print(v.name);
        #print ([v.name for v in variables_to_restore]);
        #self.saver = tf.train.Saver(var_list =variables_to_restore[1:], max_to_keep=100000);
        self.saver_official = tf.train.Saver(var_list =variables_to_restore2, max_to_keep=100000);
        self.saver = tf.train.Saver(max_to_keep=100000);

    def create_model(self, input_x, isTraining, kwargs):
        # predefined list of arguments
        args = { 'dropout':0.75};

        
        args.update(kwargs);

        # read extra argument
        dropout = float(args['dropout']);
        keep_prob = dropout;


        #with slim.arg_scope(inception_v4_arg_scope()):
        #    logits, end_points = inception_v4(input_x, num_classes=self.n_classes, is_training=isTraining)
        with slim.arg_scope(inception_v4_arg_scope(use_batch_norm=False, weight_decay=0.0005)):
            logits, end_points = inception_v4(input_x, num_classes=self.n_classes, is_training=isTraining, dropout_keep_prob=keep_prob)
        #with slim.arg_scope(inception_v4_arg_scope(use_batch_norm=False, weight_decay=0.0005)):
        #    logits, end_points = inception_v4(input_x, num_classes=self.n_classes, is_training=isTraining, dropout_keep_prob=keep_prob)

        #with slim.arg_scope(inception_v3_arg_scope()):
        #    logits, end_points = inception_v3(input_x, num_classes=self.n_classes, is_training=isTraining)
        #with slim.arg_scope(inception_v3_arg_scope()):
        #    logits, end_points = resnet_v1_50(input_x, num_classes=self.n_classes, is_training=isTraining)

    
                
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

