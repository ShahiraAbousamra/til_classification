import tensorflow as tf;

from ..sa_net_arch import AbstractCNNArch;
from ..sa_net_arch_utilities import CNNArchUtils;
from ..sa_net_optimizer import OptimizerTypes, CNNOptimizer;
from ..sa_net_data_provider import AbstractDataProvider;
import os;
import numpy as np;

class ClassifierTester:
    def __init__(self, cnn_arch:AbstractCNNArch, test_data_provider:AbstractDataProvider, session_config, output_dir, output_ext, kwargs):
        # predefined list of arguments

        self.cnn_arch = cnn_arch;
        #self.cost_func = cost_func;
        self.test_data_provider = test_data_provider;
        if(session_config == None):
            self.session_config = tf.ConfigProto();
        else:
            self.session_config = session_config;
        self.output_dir = output_dir;
        self.output_ext = output_ext;

        self.init = tf.global_variables_initializer();

    def test(self, do_init, do_restore, do_load_data):
        with tf.Session(config=self.session_config) as sess:
        #with tf.Session() as sess:
            with sess.as_default():
                if(do_init):
                    sess.run(tf.global_variables_initializer());
                    #sess.run(self.init);
                if(do_restore):
                    self.cnn_arch.restore_model(sess);
                if(do_load_data):
                    self.test_data_provider.load_data();
                    out_basefilename = self.test_data_provider.data_tag;
            

                batch_x, batch_label = self.test_data_provider.get_next_one();
                indx = 0;
                total_correct_pred = 0;
                while(batch_x is not None):
                    print(out_basefilename[indx])
                    batch_x = batch_x.reshape(1,batch_x.shape[0],batch_x.shape[1],batch_x.shape[2])
                    batch_y, correct_pred = sess.run([self.cnn_arch.logits, self.cnn_arch.correct_pred] \
                        , feed_dict={self.cnn_arch.input_x: batch_x \
                            , self.cnn_arch.labels: batch_label \
                            , self.cnn_arch.isTest: True \
                        });
                    total_correct_pred += correct_pred;

                    batch_x = self.test_data_provider.get_next_one();
                    indx = indx + 1;
                    
                print(indx);
                print(total_correct_pred);

                    

                print("Test Finished!")


