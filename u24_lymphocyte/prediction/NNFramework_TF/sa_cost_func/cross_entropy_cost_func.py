# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import sys;
import os;
import tensorflow as tf;

from ..sa_net_cost_func import AbstractCostFunc;
from ..sa_net_loss_func_helper import CNNLossFuncHelper;

class CrossEntropyCost(AbstractCostFunc):
    def __init__(self, n_classes, kwargs):
        # predefined list of arguments
        args = {'class_weights':None};

        args.update(kwargs);
        class_weights = args['class_weights'];
        if(class_weights is not None):
            class_weights = [float(x) for x in class_weights.split(',')]
        self.n_classes = n_classes;

        print('class_weights = ', class_weights);
        if(class_weights == None):
            self.class_weights = tf.Variable(tf.ones([self.n_classes]));
        else:
            self.class_weights = tf.Variable(class_weights);



    def calc_cost(self, logits, labels):
        return CNNLossFuncHelper.cost_cross_entropy(logits, labels, self.class_weights, self.n_classes);

    
