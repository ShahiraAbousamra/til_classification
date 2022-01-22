# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import sys;
import os;
import tensorflow as tf;
import numpy as np;

from .sa_net_arch_utilities import CNNArchUtils;
#from sa_net_loss_func import CNNLossFuncHelper, CostFuncTypes;
from .sa_net_cost_func import AbstractCostFunc;

class AbstractCNNArch:
    def __init__(self, n_channels, n_classes, model_out_path, model_base_filename, cost_func:AbstractCostFunc, kwargs):
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.model_out_path = model_out_path;
        self.model_base_filename = model_base_filename;
        self.current_model_checkpoint_path = None;
        self.input_x = tf.placeholder(tf.float32, [None, None, None, n_channels])
        self.labels = tf.placeholder(tf.float32, [None, None, None, n_classes])
        self.isTest = tf.placeholder(tf.bool);
        self.dropout = tf.placeholder(tf.float32);
        self.cost_func = cost_func;
        self.epochs_count = tf.Variable(0);
        self.epoch = tf.placeholder(tf.bool);
        self.dropout = tf.placeholder(tf.float32);

        self.logits, self.variables = self.create_model(self.input_x, self.isTest, self.dropout, kwargs);
        self.cost = self.cost_func.calc_cost(self.logits, self.labels);
        self.prediction_softmax = self.get_prediction_softmax(self.logits);
        self.prediction_class = self.get_class_prediction(self.logits);
        self.correct_pred = self.get_correct_prediction(self.logits, self.labels);
        self.accuracy = self.get_accuracy();
        self.saver = tf.train.Saver(max_to_keep=100000);

    def save_model(self, sess, postfix):
        filename = os.path.join(self.model_out_path, self.model_base_filename+ postfix + '.ckpt');
        self.saver.save(sess, filename );

    def restore_model(self, sess):
        #if(is_model_exist()):
        #    self.saver.restore(sess, self.model_out_path);
        #    return True;
        #return False;
        ckpt = tf.train.get_checkpoint_state(self.model_out_path);
        if ckpt and ckpt.model_checkpoint_path:
            self.current_model_checkpoint_path = ckpt.model_checkpoint_path;
            self.saver.restore(sess, ckpt.model_checkpoint_path);

    #def is_model_exist(self):
    #    if(os.path.isfile(self.model_out_path + '.ckpt.meta')):
    #        return True;
    #    return False;

    def create_model(self, input_x, isTest, dropout, kwargs):
        pass;


    #def calc_cost(self, logits, labels, kwargs):
    #    # predefined list of arguments
    #    args = {'cost_func': CostFuncTypes.CROSS_ENTROPY};
    #    args.update(kwargs);
    #    cost_type = args['cost_func'];

    #    if(cost_type == CostFuncTypes.CROSS_ENTROPY):
    #        return CNNLossFuncHelper.cost_cross_entropy(logits, labels, self.get_class_weights(logits, kwargs), self.n_classes);

    #    return 0;

    #def get_class_weights(self, logits, kwargs):
    #    ## predefined list of arguments
    #    #args = {'cost_func':'cross_entropy'};
    #    return self.class_weights;

    #def set_class_weights(self, weights, kwargs):
    #    ## predefined list of arguments
    #    #args = {'cost_func':'cross_entropy'};
    #    return self.class_weights.assign(weights);

    def get_prediction_softmax(self, logits):
        return CNNArchUtils.get_probability_softmax(logits);

    def get_class_prediction(self, logits):
        return tf.argmax(logits, axis = tf.rank(logits)-1);

    def get_correct_prediction(self, logits, labels):
        prediction = tf.argmax(logits, axis = tf.rank(logits)-1);
        label = tf.argmax(labels, axis = tf.rank(labels)-1);
        return tf.equal(prediction, label);

    def get_accuracy(self):
        return tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    #def get_prediction_size(self, sess, data_point):
    #    data_point = data_point.reshape(1, data_point.shape[0], data_point.shape[1], data_point.shape[2]);
    #    #print('data_point');
    #    #print(data_point.shape);
    #    out = sess.run([self.logits], feed_dict={self.input_x:data_point, self.isTest:True, self.dropout:0.0});
    #    #print(out);
    #    return np.shape(out);


    def get_prediction_size(self, sess, data_point):
        data_point = data_point.reshape(1, data_point.shape[0], data_point.shape[1], data_point.shape[2]);
        #tf.Print(data_point,[data_point]);
        #print(data_point.shape);
        out = sess.run([self.logits], feed_dict={self.input_x:data_point, self.isTest:True, self.dropout:0.0});
        #print('out');
        #print(np.shape(out));
        #tf.Print(out,[out]);
        shape = np.shape(out)
        out = np.reshape(out, (shape[1], shape[2], shape[3], shape[4]));
        #print(out);
        #tf.Print(out,[out]);
        #print(np.shape(out));
        return np.shape(out);