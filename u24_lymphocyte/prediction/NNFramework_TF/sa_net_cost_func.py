# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import sys;
import os;
import tensorflow as tf;

#from sa_net_loss_func import CNNLossFuncHelper;

class AbstractCostFunc:
    def __init__(self, n_classes, kwargs):
        self.n_classes = n_classes;
        self.class_weights = tf.Variable(tf.ones([n_classes]));

        self.cost = self.calc_cost(self.logits, self.labels, kwargs);

    def calc_cost(self, logits, labels):
        pass;

    
