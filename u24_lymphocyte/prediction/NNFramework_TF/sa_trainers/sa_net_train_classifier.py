# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import tensorflow as tf;
import os;
from distutils.util import strtobool;
import glob;

from ..sa_net_train import CNNTrainer;
from ..sa_net_arch import AbstractCNNArch;
from ..sa_net_arch_utilities import CNNArchUtils;
from ..sa_net_optimizer import OptimizerTypes, CNNOptimizer;
from ..sa_net_data_provider import AbstractDataProvider;
import sys 
import time
import pickle
import numpy as np

class ClassifierTrainer(CNNTrainer):
    def __init__(self, cnn_arch:AbstractCNNArch, train_data_provider:AbstractDataProvider, validate_data_provider:AbstractDataProvider, optimizer_type, session_config, kwargs):
        # predefined list of arguments
        args = {'max_epochs':1000, 'learning_rate': 0.0005, 'batch_size':256, 'epoch_size':10, 'display_step':5, 'save_best_only':'False'
            , 'subepoch_checkpoint_step':-1};
        args.update(kwargs);

        self.cnn_arch = cnn_arch;
        #self.cost_func = cost_func;
        self.train_data_provider = train_data_provider;
        self.validate_data_provider = validate_data_provider;
        self.optimizer_type = optimizer_type;
        if(session_config == None):
            self.session_config = tf.ConfigProto();
            #self.session_config.gpu_options.per_process_gpu_memory_fraction = 0.6
            self.session_config.gpu_options.allow_growth = True
            #self.session_config.log_device_placement = True
        else:
            self.session_config = session_config;
        self.max_epochs = int(args['max_epochs']);
        self.learning_rate = tf.Variable(float(args['learning_rate']));
        self.batch_size = int(args['batch_size']);
        self.epoch_size = int(args['epoch_size']);
        self.epoch_size_config = self.epoch_size ;
        self.display_step = int(args['display_step']);
        self.global_step = tf.Variable(0);
        self.save_best_only = bool(strtobool(args['save_best_only']));
        self.subepoch_checkpoint_step = int(args['subepoch_checkpoint_step']);

        self.init = tf.global_variables_initializer();
        if(self.optimizer_type == OptimizerTypes.ADAM):
            self.optimizer = CNNOptimizer.adam_optimizer(self.learning_rate, self.cnn_arch.cost, self.global_step);
        elif(self.optimizer_type == OptimizerTypes.SGD):
            self.optimizer = CNNOptimizer.sgd_optimizer(self.learning_rate, self.cnn_arch.cost, self.global_step);
        self.epoch_out_filename = os.path.join(self.cnn_arch.model_out_path, self.cnn_arch.model_base_filename + '_train_epoch_out.txt');
        self.minibatch_out_filename = os.path.join(self.cnn_arch.model_out_path, self.cnn_arch.model_base_filename + '_train_minibatch_out.txt');

    def train(self, do_init, do_restore, do_load_data):
        self.epoch_out_filewriter = open(self.epoch_out_filename, 'a+' );
        self.minibatch_out_filewriter = open(self.minibatch_out_filename, 'a+' );
        best_saved_model_filename = None;
        best_F1_saved_model_filename = None;
        best_loss_saved_model_filename = None;
        last_saved_model_filename = None;
        last_saved_subepoch_model_filename = None;

        if(do_load_data):
            self.train_data_provider.load_data();
            if(not (self.validate_data_provider is None)):
                self.validate_data_provider.load_data();            

        if(self.epoch_size < 0):
            self.epoch_size = round(self.train_data_provider.data_count / float(self.batch_size) + 0.5);
        if(not(self.validate_data_provider is None)):
            self.validate_epoch_size = round(self.validate_data_provider.data_count / float(self.batch_size) + 0.5);

        best_validation_accuracy = 0;
        best_validation_F1 = 0;
        best_validation_loss = 1000;
        best_train_val_avg_accuracy = 0;
        current_validation_accuracy = None;
        current_validation_F1 = None;
        current_validation_loss = None;
        current_train_val_avg_accuracy = None;   
        total_cost = 0;
        total_correct_count = 0;
        total_count = 0;
        step_start_num = 0; 

        with tf.Session(config=self.session_config) as sess:
            with sess.as_default():
                if(do_init):
                    sess.run(tf.global_variables_initializer());
                    #sess.run(self.init);
                if(do_restore):
                    #print('before restore');
                    self.cnn_arch.restore_model(sess);
                    epoch_start_num = self.cnn_arch.epochs_count.eval();            
                    #print('after restore');
                    if(self.subepoch_checkpoint_step > 0):
                        self.train_data_provider.restore_state(self.cnn_arch.model_restore_filename);
                        step_start_num, total_cost, total_correct_count, total_count = self.restore_state(self.cnn_arch.model_restore_filename);
                counter = 0;
                #while(True):
                #    counter +=1;
                #    if counter % 1000 == 0: 
                #        print(counter);

                for epoch in range(epoch_start_num, self.max_epochs):
                    t_sum = [0 for c in range(self.cnn_arch.n_classes)]
                    f_sum = [0 for c in range(self.cnn_arch.n_classes)]
                    for step in range(step_start_num, self.epoch_size):
                        #tic1 = time.time();
                        batch_x, batch_label = self.train_data_provider.get_next_n(self.batch_size);
                        if(batch_x is None):
                            break;
                        #print('batch_x.dtype = ', batch_x.dtype)
                        #print('min batch_x= ', np.amin(batch_x))
                        #print('max batch_x= ', np.amax(batch_x))
                        #print('batch size = ', batch_x.eval().shape[0]);
                        #print('label size = ', batch_label.eval().shape[0]);
                        #tic2 = time.time();
                        #print('time data = ', tic2 - tic1);
                        #print('batch_label.sum');
                        #print(batch_label.sum(axis=0))
                        #print('batch_label');
                        #print(batch_label)
                        #sys.stdout.flush()
                        #opt, cost, correct_pred  = sess.run([self.optimizer, self.cnn_arch.cost, self.cnn_arch.correct_pred] \
                        #        , feed_dict={self.cnn_arch.input_x: batch_x.eval() \
                        #        , self.cnn_arch.labels: batch_label.eval() \
                        #        , self.cnn_arch.isTest: False \
                        #        , self.cnn_arch.isTraining: True \
                        #    });
                        #batch_count = batch_label.eval().shape[0];

                        #tic1 = time.time();
                        opt, cost, correct_pred  = sess.run([self.optimizer, self.cnn_arch.cost, self.cnn_arch.correct_pred] \
                                , feed_dict={self.cnn_arch.input_x: batch_x \
                                , self.cnn_arch.labels: batch_label \
                                , self.cnn_arch.isTest: False \
                                , self.cnn_arch.isTraining: True \
                            });
                        #tic2 = time.time();
                        #print('time cost = ', tic2 - tic1);
                        #sys.stdout.flush()

                        batch_count = batch_label.shape[0];
                 
                        total_cost += cost;
                        batch_correct_count = correct_pred.sum();
                        total_correct_count += batch_correct_count;
                        total_count += batch_count;

                        l = np.argmax(batch_label, axis = 1);
                        #print('correct_pred.shape', correct_pred.shape)
                        for c in range(batch_label.shape[1]):            
                            t = np.logical_and(l == c, correct_pred == True);
                            f = np.logical_and(l == c, correct_pred == False);
                            t_sum[c] += t.astype(int).sum();
                            f_sum[c] += f.astype(int).sum();

                        if step % self.display_step == 0:
                            #self.output_minibatch_info(epoch, cost)
                            self.output_minibatch_info(epoch, step, cost, batch_correct_count, batch_count)

                        if(self.subepoch_checkpoint_step > 0 and step > 0 and step % self.subepoch_checkpoint_step == 0):
                            new_saved_subepoch_model_filename = self.cnn_arch.save_model(sess, self.optimizer, epoch, suffix="_step=_{:06d}".format(step));
                            self.train_data_provider.save_state(new_saved_subepoch_model_filename);
                            self.save_state(new_saved_subepoch_model_filename, step, total_cost, total_correct_count, total_count);
                            self.delete_model_files(last_saved_subepoch_model_filename);
                            last_saved_subepoch_model_filename = new_saved_subepoch_model_filename;


                    print('mean loss = ', total_cost/float(total_count))
                    print('accuracy = ', total_correct_count/float(total_count))

                    #self.output_epoch_info(epoch, total_cost);
                    self.output_epoch_info(epoch, total_cost, self.epoch_size, total_correct_count, total_count);
                    self.output_epoch_info_per_class(t_sum, f_sum);
                    current_train_accuracy = total_correct_count / float(total_count);
                        
                    
            

                    # increment number of epochs
                    sess.run(tf.assign_add(self.cnn_arch.epochs_count, 1))

                    if(not(self.validate_data_provider is None)):
                        # run in test mode to ensure batch norm is calculated based on saved mean and std
                        print("Running Validation:");
                        self.write_to_file("Running Validation"
                            , self.epoch_out_filewriter);
                        self.validate_data_provider.reset();
                        if(not(self.validate_data_provider is None)):
                            self.validate_epoch_size = round(self.validate_data_provider.data_count / float(self.batch_size) + 0.5);
                        validate_total_loss = 0;
                        validate_correct_count = 0;
                        validate_count = 0;
                        TP = 0;
                        TN = 0;
                        FP = 0;
                        FN = 0;
                        t_sum = [0 for c in range(self.cnn_arch.n_classes)]
                        f_sum = [0 for c in range(self.cnn_arch.n_classes)]
                        for validate_step in range(0, self.validate_epoch_size):
                            validate_batch_x, validate_batch_label = self.validate_data_provider.get_next_n(self.batch_size);
                            if(validate_batch_x is None):
                                break;
                            #cost, correct_pred  = sess.run([self.cnn_arch.cost, self.cnn_arch.correct_pred] \
                            #    , feed_dict={self.cnn_arch.input_x: validate_batch_x.eval() \
                            #    , self.cnn_arch.labels: validate_batch_label.eval() \
                            #    , self.cnn_arch.isTest: True \
                            #});
                            #validate_count += validate_batch_label.eval().shape[0];
                 
                            cost, correct_pred  = sess.run([self.cnn_arch.cost, self.cnn_arch.correct_pred] \
                                , feed_dict={self.cnn_arch.input_x: validate_batch_x \
                                , self.cnn_arch.labels: validate_batch_label \
                                , self.cnn_arch.isTest: True \
                                , self.cnn_arch.isTraining: False \
                            });
                            validate_count += validate_batch_label.shape[0];

                            validate_total_loss += cost;
                            validate_correct_count += correct_pred.sum();

                            l = np.argmax(validate_batch_label, axis = 1);
                            for c in range(validate_batch_label.shape[1]):
                                t = np.logical_and(l == c, correct_pred == True);
                                f = np.logical_and(l == c, correct_pred == False);
                                t_sum[c] += t.astype(int).sum();
                                f_sum[c] += f.astype(int).sum();
                                
                            TP += np.logical_and(l == 1, correct_pred == True).astype(int).sum();
                            TN += np.logical_and(l == 0, correct_pred == True).astype(int).sum();
                            FN += np.logical_and(l == 1, correct_pred == False).astype(int).sum();
                            FP += np.logical_and(l == 0, correct_pred == False).astype(int).sum();

                            #break;
                        
                        current_validation_accuracy = validate_correct_count / float(validate_count);
                        current_train_val_avg_accuracy = (current_train_accuracy + current_validation_accuracy)/2.0;
                        val_precision = TP / float(TP + FP);
                        val_recall = TP / float(TP + FN);
                        val_f1 = 2 / (1/val_precision + 1/val_recall);
                        val_loss = validate_total_loss/ float(self.validate_epoch_size);
                        self.output_epoch_info(epoch, validate_total_loss, self.validate_epoch_size, validate_correct_count, validate_count);                        
                        self.output_epoch_info_per_class(t_sum, f_sum);
                        self.output_epoch_info_F1(TP, TN, FP, FN, val_precision, val_recall, val_f1);
                        #self.output_epoch_info(epoch, validate_total_loss, self.validate_epoch_size, validate_correct_count, validate_count);                        
                        #print('validation_accuracy = ', current_validation_accuracy);


                      #  if((not self.save_best_only)  
		                    #or (current_validation_accuracy is None) 
		                    #or (current_validation_accuracy >= best_validation_accuracy) \
		                    #or (current_train_val_avg_accuracy >= best_train_val_avg_accuracy) \
		                    #):
                        saved = False;
                        if(val_f1 > best_validation_F1):
	                        #self.cnn_arch.save_model(sess, '_epoch_' + str(epoch));
                            best_validation_F1 = val_f1;
                            #best_train_val_avg_accuracy = current_train_val_avg_accuracy;
                            print("Saving model:");
                            new_best_F1_saved_model_filename = self.cnn_arch.save_model(sess, self.optimizer, epoch, suffix="_F1");
                            self.delete_model_files(best_F1_saved_model_filename);
                            best_F1_saved_model_filename = new_best_F1_saved_model_filename;
                            saved = True;
                        if((current_validation_accuracy is None) 
		                    or (current_validation_accuracy > best_validation_accuracy) \
		                    ):
	                        #self.cnn_arch.save_model(sess, '_epoch_' + str(epoch));
                            best_validation_accuracy = current_validation_accuracy;
                            #best_train_val_avg_accuracy = current_train_val_avg_accuracy;
                            print("Saving model:");
                            new_best_saved_model_filename = self.cnn_arch.save_model(sess, self.optimizer, epoch, suffix="_accuracy");
                            self.delete_model_files(best_saved_model_filename);
                            best_saved_model_filename = new_best_saved_model_filename;
                            saved = True;
                        if(val_loss < best_validation_loss):
                            best_validation_loss = val_loss;
                            print("Saving model:");
                            new_best_loss_saved_model_filename = self.cnn_arch.save_model(sess, self.optimizer, epoch, suffix="_loss");
                            self.delete_model_files(best_loss_saved_model_filename);
                            best_loss_saved_model_filename = new_best_loss_saved_model_filename;
                            saved = True;
                        if(not saved or not self.save_best_only):
                            new_saved_model_filename = self.cnn_arch.save_model(sess, self.optimizer, epoch);
                            if(self.save_best_only):
                                self.delete_model_files(last_saved_model_filename);
                            last_saved_model_filename = new_saved_model_filename;

		                # permute the training data for the next epoch
                        self.train_data_provider.reset(repermute=True);
                        if(self.epoch_size_config < 0):
                            self.epoch_size = round(self.train_data_provider.data_count / float(self.batch_size) + 0.5);


                    total_cost = 0;
                    total_correct_count = 0;
                    total_count = 0;
                    step_start_num = 0; 

                print("Optimization Finished!")


#    def output_minibatch_info(self, epoch, cost):
#        print("epoch = " + str(epoch) \
#            + ", minibatch loss= " + "{:.6f}".format(cost) \
#        );

#    def output_epoch_info(self, epoch, total_cost):
#        print("\r\nepoch = " + str(epoch) \
#            + ", total loss= " + "{:.6f}".format(total_cost) \
#        );

    def output_minibatch_info(self, epoch, batch, cost, correct_count, total_count):
        print("epoch = " + str(epoch) \
            + ", batch# = " + str(batch) \
            + ", minibatch loss= " + "{:.6f}".format(cost) \
            + ", correct count= " + "{:d}".format(correct_count) \
            + ", total count= " + "{:d}".format(total_count) \
            + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
            + ", time=" + str(time.asctime(time.localtime())) \
            + ", tlapse= " + str(time.process_time()) \
        );
        sys.stdout.flush()
        self.write_to_filename("epoch = " + str(epoch) \
            + ", batch# = " + str(batch) \
            + ", minibatch loss= " + "{:.6f}".format(cost) \
            + ", correct count= " + "{:d}".format(correct_count) \
            + ", total count= " + "{:d}".format(total_count) \
            + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
            + ", time=" + str(time.asctime(time.localtime())) \
            + ", tlapse= " + str(time.process_time()) \
            , self.minibatch_out_filename
        );


    def output_epoch_info(self, epoch, total_cost, n_batches, correct_count, total_count):
        print("\r\nepoch = " + str(epoch) \
            + ", avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", correct count= " + "{:d}".format(correct_count) \
            + ", total count= " + "{:d}".format(total_count) \
            + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
        );
        sys.stdout.flush()
        self.write_to_filename("epoch = " + str(epoch) \
            + ", avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", correct count= " + "{:d}".format(correct_count) \
            + ", total count= " + "{:d}".format(total_count) \
            + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
            + ", time=" + str(time.asctime(time.localtime())) \
            + ", tlapse= " + str(time.process_time()) \
            , self.epoch_out_filename
        );
        self.write_to_filename("\r\n epoch = " + str(epoch) \
            + ", avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", correct count= " + "{:d}".format(correct_count) \
            + ", total count= " + "{:d}".format(total_count) \
            + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
            + ", time=" + str(time.asctime(time.localtime())) \
            + ", tlapse= " + str(time.process_time()) \
            , self.minibatch_out_filename
        );

    def output_epoch_info_per_class(self, t_sum, f_sum):
        for c in range(len(t_sum)):
            print('class ', c, ': Correct = ', t_sum[c], ' Wrong = ', f_sum[c], '\r\n' );
            self.write_to_filename('class ' + str(c) + ': Correct = ' + str(t_sum[c]) + ' Wrong = ' + str(f_sum[c]) \
                , self.epoch_out_filename
            );
            self.write_to_filename('class ' + str(c) + ': Correct = ' + str(t_sum[c]) + ' Wrong = ' + str(f_sum[c]) \
                , self.minibatch_out_filename
            );
        sys.stdout.flush()

    def output_epoch_info_F1(self, TP, TN, FP, FN, precision, recall, F1):
        print('TP = ', TP, ', TN = ', TN, ', FP = ', FP, ', FN = ', FN, '\r\n' );
        self.write_to_filename('TP = ' + str(TP) + ', TN = ' + str(TN) + ', FP = ' + str(FP) + ', FN = ' + str(FN) \
            , self.epoch_out_filename
        );
        self.write_to_filename('precision = ' + str(precision) + ', recall = ' + str(recall) + ', F1 = ' + str(F1)  \
            , self.epoch_out_filename
        );
        self.write_to_filename('TP = ' + str(TP) + ', TN = ' + str(TN) + ', FP = ' + str(FP) + ', FN = ' + str(FN) \
            , self.minibatch_out_filename
        );
        self.write_to_filename('precision = ' + str(precision) + ', recall = ' + str(recall) + ', F1 = ' + str(F1)  \
            , self.minibatch_out_filename
        );
        sys.stdout.flush()

    def write_to_file(self, text, filewriter):
        filewriter.write('\r\n');
        filewriter.write(text);
        filewriter.flush();

    def write_to_filename(self, text, filename):
        filewriter = open(filename, 'a+' );
        filewriter.write('\r\n');
        filewriter.write(text);
        filewriter.flush();
        filewriter.close()



#    def print_optimizer_params(self):
#        # Print optimizer's state_dict
#        print("Optimizer's state_dict:")
#        for var_name in self.optimizer.state_dict():
#            print(var_name, "\t", self.optimizer.state_dict()[var_name])

    def delete_model_files(self, filepath):
        if(filepath is None):
            return;
        filepath, _ = os.path.splitext(filepath);
        print('delete_model_files = ', filepath)
        file_pattern = filepath + '*';
        files = glob.glob(file_pattern);
        for file in files: 
            print(file);
            os.remove(file);


    def save_state(self, checkpoint_filepath, step, total_cost, total_correct_count, total_count):
        if(checkpoint_filepath is None):
            return;
        base_filename,_ = os.path.splitext(checkpoint_filepath);
        filepath_param = base_filename + '_train_out_params.pkl' ;
        pickle.dump([step, total_cost, total_correct_count, total_count], open(filepath_param, 'wb'));

    def restore_state(self, checkpoint_filepath):
        if(checkpoint_filepath is None):
            return 0, 0, 0, 0;
        base_filename,_ = os.path.splitext(checkpoint_filepath);
        filepath_param = base_filename + '_train_out_params.pkl' ;
        if(os.path.isfile(filepath_param)):            
            step, total_cost, total_correct_count, total_count = pickle.load(open(filepath_param, 'rb'));
        else:
            step, total_cost, total_correct_count, total_count = 0, 0, 0, 0;

        print('step, total_cost, total_correct_count, total_count = ', step, total_cost, total_correct_count, total_count);
        return step, total_cost, total_correct_count, total_count;