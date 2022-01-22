# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import tensorflow as tf;

from .sa_net_arch import AbstractCNNArch;
from .sa_net_arch_utilities import CNNArchUtils;
from .sa_net_optimizer import OptimizerTypes, CNNOptimizer;
from .sa_net_data_provider import AbstractDataProvider;


class CNNTrainer:
    def __init__(self, cnn_arch:AbstractCNNArch, train_data_provider:AbstractDataProvider, optimizer_type, session_config, kwargs):
        # predefined list of arguments
        args = {'max_epochs':1000, 'learning_rate': 0.0005, 'batch_size':256, 'epoch_size':10};
        args.update(kwargs);

        self.cnn_arch = cnn_arch;
        #self.cost_func = cost_func;
        self.train_data_provider = train_data_provider;
        self.optimizer_type = optimizer_type;
        if(session_config == None):
            self.session_config = tf.ConfigProto();
        else:
            self.session_config = session_config;
        self.max_epochs = int(args['max_epochs']);
        self.learning_rate = tf.Variable(float(args['learning_rate']));
        self.batch_size = int(args['batch_size']);
        self.epoch_size = int(args['epoch_size']);
        self.global_step = tf.Variable(0);

        self.init = tf.global_variables_initializer();
        if(self.optimizer_type == OptimizerTypes.ADAM):
            self.optimizer = CNNOptimizer.adam_optimizer(self.learning_rate, self.cnn_arch.cost, self.global_step);

    def train(self, do_init, do_restore, do_load_data, display_step=5):
        with tf.Session(config=self.session_config) as sess:
        #with tf.Session() as sess:
            with sess.as_default():
                if(do_init):
                    sess.run(tf.global_variables_initializer());
                    #sess.run(self.init);
                if(do_restore):
                    self.cnn_arch.restore_model(sess);
                if(do_load_data):
                    self.train_data_provider.load_data();
            
                epoch_start_num = self.cnn_arch.epochs_count.eval();            

                tester_x, tester_label = self.train_data_provider.get_next_one();
                out_shape = self.cnn_arch.get_prediction_size(sess, tester_x);

                for epoch in range(epoch_start_num, self.max_epochs):
                    total_cost = 0;
                    for step in range(0, self.epoch_size):
                        batch_x, batch_label = self.train_data_provider.get_next_n(self.batch_size);
                        batch_label = CNNArchUtils.crop_to_shape(batch_label, out_shape);
                        opt, cost, accuracy = sess.run([self.optimizer, self.cnn_arch.cost, self.cnn_arch.accuracy] \
                            , feed_dict={self.cnn_arch.input_x: batch_x \
                                , self.cnn_arch.labels: batch_label \
                                , self.cnn_arch.isTest: False \
                            });
                 
                        if step % display_step == 0:
                            self.output_minibatch_info(epoch, cost, accuracy)
                        
                        total_cost += cost;

                    # increment number of epochs
                    sess.run(tf.assign_add(self.cnn_arch.epochs_count, 1))

                    self.output_epoch_info(epoch, total_cost);
                        
                    
                    self.cnn_arch.save_model(sess, '_epoch_' + str(epoch));
            

                print("Optimization Finished!")


    def output_minibatch_info(self, epoch, cost, accuracy):
        print("epoch = " + str(epoch) \
            + ", minibatch loss= " + "{:.6f}".format(cost) \
            + ", accuracy= " + "{:.6f}".format(accuracy) \
        );

    def output_epoch_info(self, epoch, total_cost):
        print("\r\nepoch = " + str(epoch) \
            + ", total loss= " + "{:.6f}".format(total_cost) \
        );
