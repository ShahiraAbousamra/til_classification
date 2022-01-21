import tensorflow as tf;

from ..sa_net_arch import AbstractCNNArch;
from ..sa_net_arch_utilities import CNNArchUtils;
from ..sa_net_optimizer import OptimizerTypes, CNNOptimizer;
from ..sa_net_data_provider import AbstractDataProvider;
import os;
import numpy as np;
import time

class ClassifierTesterBatch:
    def __init__(self, cnn_arch:AbstractCNNArch, test_data_provider:AbstractDataProvider, session_config, output_dir, output_ext, kwargs):
        # predefined list of arguments
        args = {'batch_size':1, 'stats_file_suffix':'_test_stats_out.txt', 'predictions_file_suffix':'_test_pred_out.csv', 'threshold':0.5};
        args.update(kwargs);

        self.cnn_arch = cnn_arch;
        #self.cost_func = cost_func;
        self.test_data_provider = test_data_provider;
        if(session_config == None):
            self.session_config = tf.ConfigProto();
        else:
            self.session_config = session_config;
        self.output_dir = output_dir;
        self.output_ext = output_ext;
        self.batch_size = int(args['batch_size']);
        self.threshold = float(args['threshold']);
        #self.stats_file_suffix = str(args['stats_file_suffix']);
        #self.predictions_file_suffix = str(args['predictions_file_suffix']);
        #self.test_out_filename = os.path.join(self.output_dir, self.cnn_arch.model_base_filename + self.stats_file_suffix);
        #self.predictions_filename = os.path.join(self.output_dir, self.cnn_arch.model_base_filename + self.predictions_file_suffix);
        #self.labels_np_filename = os.path.join(self.output_dir, self.cnn_arch.model_base_filename + "_labels.npy");
        #self.predictions_np_filename = os.path.join(self.output_dir, self.cnn_arch.model_base_filename + "_predictions.npy");

        self.init = tf.global_variables_initializer();

    def test(self, do_init, do_restore, do_load_data):
        self.predictions = [];
        self.predictions_binary = [];
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
                    #out_basefilename = self.test_data_provider.data_tag;
            

                #batch_x, batch_label = self.test_data_provider.get_next_one();
                #indx = 0;
                #total_correct_pred = 0;
                #while(batch_x is not None):
                #    print(out_basefilename[indx])
                #    batch_x = batch_x.reshape(1,batch_x.shape[0],batch_x.shape[1],batch_x.shape[2])
                #    batch_y, correct_pred = sess.run([self.cnn_arch.logits, self.cnn_arch.correct_pred] \
                #        , feed_dict={self.cnn_arch.input_x: batch_x \
                #            , self.cnn_arch.labels: batch_label \
                #            , self.cnn_arch.isTest: True \
                #        });
                #    total_correct_pred += correct_pred;

                #    batch_x = self.test_data_provider.get_next_one();
                #    indx = indx + 1;
                    

                self.epoch_size = round(self.test_data_provider.data_count / float(self.batch_size) + 0.5);
                count = 0;
                #total_loss = 0;
                #correct_count = 0;
                #TP = 0;
                #TN = 0;
                #FP = 0;
                #FN = 0;
                #t_sum = [0 for c in range(self.cnn_arch.n_classes)]
                #f_sum = [0 for c in range(self.cnn_arch.n_classes)]
                print('start time = ', str(time.asctime(time.localtime())));
                for step in range(0, self.epoch_size):
                    print('step = ', step)
                    batch_x, batch_label = self.test_data_provider.get_next_n(self.batch_size);
                    if(batch_x is None):
                        break;
                    #print('batch_x.shape = ', batch_x.shape)
                    #cost, correct_pred  = sess.run([self.cnn_arch.cost, self.cnn_arch.correct_pred] \
                    #    , feed_dict={self.cnn_arch.input_x: validate_batch_x.eval() \
                    #    , self.cnn_arch.labels: validate_batch_label.eval() \
                    #    , self.cnn_arch.isTest: True \
                    #});
                    #validate_count += validate_batch_label.eval().shape[0];
                 
                    batch_y  = sess.run([self.cnn_arch.logits] \
                        , feed_dict={self.cnn_arch.input_x: batch_x \
                        , self.cnn_arch.isTest: True \
                        , self.cnn_arch.isTraining: False \
                    });
                    batch_y = np.array(batch_y);
                    #print('batch_y.shape = ', batch_y.shape)
                    #print('batch_y = ', batch_y)
                    #print('batch_y[..., -1:] = ', batch_y[..., -1:].squeeze())
                    count += batch_x.shape[0];
                    batch_y_sig = self.sigmoid(batch_y[..., -1:].squeeze());
                    #print('batch_y_sig = ', batch_y_sig)
                    self.predictions.append(batch_y_sig);
                    batch_y_binary = batch_y_sig > self.threshold
                    #print('batch_y_binary = ', batch_y_binary)
                    self.predictions_binary.append(batch_y_binary);

                    #print(self.sigmoid(batch_y));
                    #total_loss += cost;
                    #correct_count += correct_pred.sum();

                    #l = np.argmax(batch_label, axis = 1);
                    #for c in range(batch_label.shape[1]):
                    #    t = np.logical_and(l == c, correct_pred == True);
                    #    f = np.logical_and(l == c, correct_pred == False);
                    #    t_sum[c] += t.astype(int).sum();
                    #    f_sum[c] += f.astype(int).sum();
                                
                    #TP += np.logical_and(l == 1, correct_pred == True).astype(int).sum();
                    #TN += np.logical_and(l == 0, correct_pred == True).astype(int).sum();
                    #FN += np.logical_and(l == 1, correct_pred == False).astype(int).sum();
                    #FP += np.logical_and(l == 0, correct_pred == False).astype(int).sum();

                    #self.output_predictions(batch_y, l, correct_pred, self.test_data_provider.datapoints_files_list);
                    
                    #if(step == 2):
                    #    break;
                        
                #accuracy = correct_count / float(count);
                #precision = TP / float(TP + FP);
                #recall = TP / float(TP + FN);
                #f1 = 2 / (1/precision + 1/recall);
                #self.output_stats(total_loss, self.epoch_size, correct_count, count);                        
                #self.output_stats_per_class(t_sum, f_sum);
                #self.output_stats_F1(TP, TN, FP, FN, precision, recall, f1);
                print('end time = ', str(time.asctime(time.localtime())));

                self.write_predictions();

                print("Test Finished!")

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def write_predictions(self):
        filepath = os.path.join(self.output_dir, self.cnn_arch.model_base_filename + '_pred_prob.npy');
        print('np.array(self.predictions).reshape((-1)).shape = ', np.array(self.predictions).reshape((-1)).shape);
        np.array(self.predictions).reshape((-1)).dump(filepath);
        filepath = os.path.join(self.output_dir, self.cnn_arch.model_base_filename + '_pred_binary.npy');
        np.array(self.predictions_binary).astype(np.float).reshape((-1)).dump(filepath);
        self.test_data_provider.write_label_info(self.output_dir, self.cnn_arch.model_base_filename)

    #def output_predictions(self, batch_y, batch_label, correct_pred, datapoints_files_list):

    #    filewriter = open(self.predictions_filename, 'a+' );
    #    batch_y_max = np.argmax(batch_y, axis=-1)
    #    batch_y_sigmoid = self.sigmoid(batch_y)
    #    for i in range(batch_y.shape[0]):
    #        filewriter.write(datapoints_files_list[i] + ',' + str(batch_label[i]) + ',' + str(batch_y_max[i]) + ',' + str(correct_pred[i]) + ','+str(batch_y_sigmoid[i,:]));
    #        filewriter.write('\r\n');

    #    filewriter.flush();
    #    filewriter.close()

    #    self.labels.append(batch_label[i]);
    #    self.predictions.append(batch_y_sigmoid[i,1]);
            

    #def output_stats(self, total_cost, n_batches, correct_count, total_count):
    #    self.write_to_filename( 
    #        ", avg loss= " + "{:.6f}".format(total_cost / float(n_batches)) \
    #        + ", correct count= " + "{:d}".format(correct_count) \
    #        + ", total count= " + "{:d}".format(total_count) \
    #        + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
    #        , self.test_out_filename
    #    );

    #def output_stats_per_class(self, t_sum, f_sum):
    #    for c in range(len(t_sum)):
    #        self.write_to_filename('class ' + str(c) + ': Correct = ' + str(t_sum[c]) + ' Wrong = ' + str(f_sum[c]) \
    #            , self.test_out_filename
    #        );

    #def output_stats_F1(self, TP, TN, FP, FN, precision, recall, F1):
    #    self.write_to_filename('TP = ' + str(TP) + ', TN = ' + str(TN) + ', FP = ' + str(FP) + ', FN = ' + str(FN) \
    #        , self.test_out_filename
    #    );
    #    self.write_to_filename('precision = ' + str(precision) + ', recall = ' + str(recall) + ', F1 = ' + str(F1)  \
    #        , self.test_out_filename
    #    );


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
