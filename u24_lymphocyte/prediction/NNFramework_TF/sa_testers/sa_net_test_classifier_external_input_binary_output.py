import tensorflow as tf;

from ..sa_net_arch import AbstractCNNArch;
from ..sa_net_arch_utilities import CNNArchUtils;
from ..sa_net_optimizer import OptimizerTypes, CNNOptimizer;
from ..sa_net_data_provider import AbstractDataProvider;
import os;
import numpy as np;

class ClassifierTesterExternalInputBinaryOutput:
    def __init__(self, cnn_arch:AbstractCNNArch, session_config, output_dir, output_ext, kwargs):
        # predefined list of arguments
        args = {'threshold':0.5};
        args.update(kwargs);

        self.cnn_arch = cnn_arch;
        #self.cost_func = cost_func;
        #self.test_data_provider = test_data_provider;
        if(session_config == None):
            self.session_config = tf.ConfigProto();
            self.session_config.gpu_options.allow_growth = True
        else:
            self.session_config = session_config;
        self.output_dir = output_dir;
        self.output_ext = output_ext;
        self.threshold = float(args['threshold']);

        self.init = tf.global_variables_initializer();

    def init_model(self, do_init, do_restore):
        self.sess = tf.Session(config=self.session_config);
        with self.sess.as_default():
            if(do_init):
                self.sess.run(tf.global_variables_initializer());
                #sess.run(self.init);
            if(do_restore):
                self.cnn_arch.restore_model(self.sess);


    def predict(self, inputs):
        with self.sess.as_default():
            batch_x = inputs;
            if (batch_x is None):
                return None;

            batch_x = self.preprocess_input(inputs);

            batch_y = self.sess.run([self.cnn_arch.logits] \
                , feed_dict={self.cnn_arch.input_x: batch_x \
                    , self.cnn_arch.isTest: True \
                });
            batch_y_sig = self.sigmoid(np.array(batch_y)[...,-1]);
            batch_y_binary = np.array(batch_y_sig > self.threshold).astype(np.float).reshape((-1,1));

            return batch_y_binary;


    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def restart_model(self):
        self.sess.close();
        #tf.reset_default_graph();
        self.init_model(True, True);

    def preprocess_input(self, inputs):
        # normalize (mean 0, std=2)
        np.clip(inputs, 0, 255, inputs);
        inputs /= 255;
        inputs -= 0.5;
        inputs *= 2;
        inputs = tf.image.resize_images(inputs, (self.cnn_arch.input_img_height, self.cnn_arch.input_img_width));
        inputs = inputs.eval()
        return inputs;
