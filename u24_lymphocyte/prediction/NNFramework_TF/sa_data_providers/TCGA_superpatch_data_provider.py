from ..sa_net_data_provider import AbstractDataProvider;
from numpy import random;
import glob

#import scipy.io as spio;
import numpy as np;
import glob;
import os;
import pickle;
from distutils.util import strtobool;
import tensorflow as tf;
from skimage import io;
from skimage import transform as sktransform;
import math;
import sys;
import time;

class TCGASuperpatchBatchDataProvider(AbstractDataProvider):
    def __init__(self, is_test, filepath_data, filepath_label, n_channels, n_classes, do_preprocess, do_augment, data_var_name=None, label_var_name=None, permute=False, repeat=True, kwargs={}):
        #super(MatDataProvider, self).__init__(filepath_data, filepath_label, n_channels, n_classes);
        args = {'input_img_height':460, 'input_img_width': 700, 'file_name_suffix':'', 'pre_resize':'False', 'postprocess':'False'};
        args.update(kwargs);    
        self.input_img_height = int(args['input_img_height']);
        self.input_img_width = int(args['input_img_width']);
        #print('self.input_img_width, self.input_img_height = ', self.input_img_width, self.input_img_height);
        self.file_name_suffix = args['file_name_suffix'];
        self.pre_resize = bool(strtobool(args['pre_resize']));
        self.do_postprocess = bool(strtobool(args['postprocess']));
        self.is_test = is_test; # note that label will be None when is_test is true
        self.filepath_data = filepath_data;
        if(filepath_label == None or filepath_label.strip() == ''):
            self.filepath_label = None ;
        else:
            self.filepath_label = filepath_label ;
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.do_preprocess = do_preprocess;
        self.do_augment = do_augment;
        self.data_var_name = data_var_name;
        self.label_var_name = label_var_name;
        self.do_permute = permute;
        self.do_repeat = repeat;
        if(do_augment):
            self.create_augmentation_map(kwargs);
        if(self.do_postprocess):
            self.read_postprocess_parameters(kwargs); 
        if(self.do_preprocess):
            self.read_preprocess_parameters(kwargs); 

        self.is_loaded = False;
        self.tmp_index = 0;

        self.in_size_x = None;
        self.in_size_y = None;

        self.n_subpatches_per_side = 8;

        if(do_augment):
            self.tf_data_points = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None))
            self.tf_translations = tf.placeholder(dtype=tf.float32, shape=(None, None))
            self.tf_angles = tf.placeholder(dtype=tf.float32, shape=(None,))
            self.tf_post_crop_y1 = tf.placeholder(dtype=tf.int32)
            self.tf_post_crop_x1 = tf.placeholder(dtype=tf.int32)
            #print('aug_hue = ', self.aug_hue_max/255.0)

            ###### TODO: Make augmentation configured - not preset
            self.tf_data_points_tmp1 = tf.image.random_flip_left_right(self.tf_data_points);
            self.tf_data_points_tmp2 = tf.image.random_flip_up_down(self.tf_data_points_tmp1);
            self.tf_data_points_tmp3 = tf.contrib.image.rotate(self.tf_data_points_tmp2, self.tf_angles);
            self.tf_data_points_tmp4 = tf.image.random_hue(self.tf_data_points_tmp3, self.aug_hue_max/255.0);
            self.tf_data_points_tmp5 = tf.image.random_brightness(self.tf_data_points_tmp4, float(self.aug_brightness_max));
            self.tf_data_points_tmp6 = tf.contrib.image.translate(self.tf_data_points_tmp5, self.tf_translations);
            self.tf_data_points_tmp7 = tf.image.crop_to_bounding_box(self.tf_data_points_tmp6, self.tf_post_crop_y1, self.tf_post_crop_x1, self.post_crop_height, self.post_crop_width);
            self.tf_data_points_aug = tf.image.resize_images(self.tf_data_points_tmp7, (self.input_img_height, self.input_img_width));

        self.tf_data_points_std_in = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None))
        self.tf_data_points_std = tf.map_fn(lambda img:tf.image.per_image_standardization(img), self.tf_data_points_std_in)


    def load_data(self):
        self.data = None;
        self.label = None;
        self.last_fetched_indx = -1;
        self.permutation = None;
        self.data_count = 0;
        self.data = None;
        self.labels = None;

        self.cancer_type_list = [];
        self.filename_list = [];
        self.individual_labels_list = []
        self.avg_label_list = []
        self.pred_old_list = []

        # read csv file
        with open(os.path.join(self.filepath_label), 'r') as label_file:
            line = label_file.readline();
            line = label_file.readline();
            while(line):
                c, s, p, i1, i2, i3, i4, i5, i6, pred_old= line.split(',');
                if (i1.strip()==""):
                    i1 = 0;
                if (i2.strip()==""):
                    i2 = 0;
                if (i3.strip()==""):
                    i3 = 0;
                if (i4.strip()==""):
                    i4 = 0;
                if (i5.strip()==""):
                    i5 = 0;
                if (i6.strip()==""):
                    i6 = 0;
                self.cancer_type_list.append(c);
                self.filename_list.append(s+'_'+p+'.png');
                self.individual_labels_list.append([int(i1), int(i2), int(i3), int(i4), int(i5), int(i6)]);
                self.avg_label_list.append(np.mean(np.array([float(i1), float(i2), float(i3), float(i4), float(i5), float(i6)])));
                self.pred_old_list.append(pred_old);
                line = label_file.readline();

        self.data = self.filename_list;
        self.data_count = len(self.data);

        ### Load data file
        #for batch_id in range(1, 6):
        #    filename = os.path.join(self.filepath_data, 'preprocess_batch_' + str(batch_id) + '.p')
        #    features, labels = pickle.load(open(filename, mode='rb'))
        #    if(batch_id == 1):
        #        self.data = features;
        #        self.labels = labels;
        #    else:
        #        self.data = np.append(self.data, features, axis=0);
        #        self.labels = np.append(self.labels, labels, axis=0);

        #self.data_count = self.data.shape[0];
        #print('data_count', self.data_count)
        #print(self.data.shape)


        ## Permutation
        if(self.do_permute == True):
            self.permutation = np.random.permutation(self.data_count)
        else:
            self.permutation = None;

        self.is_loaded = True;
        print('self.last_fetched_indx, self.data_count = ', self.last_fetched_indx, self.data_count)


    def reset(self, repermute=None):
        self.last_fetched_indx = -1;
        if(repermute == True):
            self.load_data();
            self.do_permute = True;
            self.permutation = np.random.permutation(self.data_count);

    def get_next_one(self):
        ## make sure data is loaded first
        if(self.is_loaded == False):
            self.load_data();

        ## get next data point and its corresponding label
        self.last_fetched_indx = (self.last_fetched_indx + 1);
        if(self.do_repeat == False):
            if (self.last_fetched_indx >= self.data_count):
                if(self.filepath_label == None):
                    return None;
                else:
                    return None, None;
        else:
            self.last_fetched_indx = self.last_fetched_indx % self.data_count;
        actual_indx = self.last_fetched_indx ;
        if(self.permutation is not None):
            actual_indx = self.permutation[self.last_fetched_indx];
        #data_point = self.data[actual_indx, :,:,:];
        #if(self.filepath_label == None):
        #    label = None;
        #else:
        #    label = self.labels[actual_indx,:];
        data_point, label = self.load_datapoint(actual_indx);

        ## process the data
        if(self.do_preprocess == True):
            data_point, label = self.preprocess(data_point, label);
    
       
        #if(self.do_augment == True):
        #    data_point, label = self.augment(data_point, label);

        #if(self.do_postprocess):
        #    #data_point, label = self.postprocess(data_point, label);
        #    data_point, label = tf.py_func(self.postprocess, [data_point, label], [tf.uint8, tf.int8]); 

        ## normalize [0,1]
        #data_point /= data_point.max();
        ## normalize [-1,1]        
        #data_point = tf.subtract(data_point, 0.5);
        #data_point = tf.multiply(data_point, 2.0);
        
        #data_point = tf.image.per_image_standardization(data_point);

        if(self.filepath_label == None):
            #print("return 1")
            return data_point;
        else:
            #print("return 2")
            return data_point, label;

    ## returns None, None if there is no more data to retrieve and repeat = false
    def get_next_n_old(self, n:int):
        ## validate parameters
        if(n <= 0):
            return None, None;

        ## make sure data is loaded first
        if(self.is_loaded == False):
            self.load_data();

        ## Get number of data points to retrieve        
        if(self.do_repeat == False):
            if (self.last_fetched_indx + n >= self.data_count):
                n = self.data_count - self.last_fetched_indx - 1;
                if(n <= 0):
                    return None, None;

        ## Get data shape
        data_size_x = self.input_img_width;
        data_size_y = self.input_img_height;    

        ##print(data_size_x, data_size_y);
        #data_points = tf.zeros((n, data_size_x, data_size_y, self.n_channels))
        ##print(data_points);
        #if(self.filepath_label is None):
        #    data_labels = None;
        #else:
        #    data_labels = tf.zeros((n, self.n_classes))

        datapoints_list = [];
        if(self.filepath_label is None):
            data_labels = None;
            labels_list = None;
        else:
            labels_list = [];
        
    
        for i in range(0, n):
            if(labels_list is None):
                #data_points[i] = self.get_next_one();
                datapoints_list.append(self.get_next_one());
            else:
                #data_points[i], data_labels[i] = self.get_next_one();
                data_point_tmp, data_label_tmp = self.get_next_one();
                datapoints_list.append(data_point_tmp);
                labels_list.append(data_label_tmp);

        data_points = tf.stack(datapoints_list);
        #data_points = np.stack(datapoints_list);
        if(labels_list is not None):
            data_labels = tf.stack(labels_list);
            #data_labels = np.stack(labels_list);
        #print('data_points =' + str(np.shape(data_points)))
        sys.stdout.flush();

        if(self.do_augment == True):
            data_points, data_labels = self.augment_batch(data_points, data_labels);

        if(self.do_postprocess):
            data_points, data_labels = self.postprocess_batch(data_points, data_labels);

        data_points = tf.map_fn(lambda img:tf.image.per_image_standardization(img), data_points)

        return data_points, data_labels;

    def get_next_n_old2(self, n:int):
        #print('self.last_fetched_indx = ', self.last_fetched_indx);
        ## validate parameters
        if(n <= 0):
            return None, None;

        ## make sure data is loaded first
        if(self.is_loaded == False):
            self.load_data();

        ## Get number of data points to retrieve        
        if(self.do_repeat == False):
            if (self.last_fetched_indx + n >= self.data_count):
                n = self.data_count - self.last_fetched_indx - 1;
                if(n <= 0):
                    return None, None;

        ## Get data shape
        data_size_x = self.input_img_width;
        data_size_y = self.input_img_height;    

        ##print(data_size_x, data_size_y);
        #data_points = np.zeros((n, data_size_x, data_size_y, self.n_channels))
        data_points = np.zeros((n, 300, 300, self.n_channels))
        ##print(data_points);
        if(self.filepath_label is None):
            data_labels = None;
        else:
            data_labels = np.zeros((n, self.n_classes))

        datapoints_list = [];
        if(self.filepath_label is None):
            data_labels = None;
            labels_list = None;
        else:
            labels_list = [];
        
    
        for i in range(0, n):
            if(labels_list is None):
                #data_points[i] = self.get_next_one();
                datapoints_list.append(self.get_next_one());
            else:
                #data_points[i], data_labels[i] = self.get_next_one();
                data_points[i, :,:,:], data_labels[i,:] = self.get_next_one();
                #datapoints_list.append(data_point_tmp);
                #labels_list.append(data_label_tmp);

        #data_points = tf.stack(datapoints_list);
        #data_points = np.stack(datapoints_list);
        #if(labels_list is not None):
        #    #data_labels = tf.stack(labels_list);
        #    data_labels = np.stack(labels_list);
        #print('data_points =' + str(np.shape(data_points)))
        sys.stdout.flush();

        if(self.do_augment == True):
            data_points, data_labels = self.augment_batch_new(data_points, data_labels);

        if(self.do_postprocess):
            data_points, data_labels = self.postprocess_batch_new(data_points, data_labels);

        data_points = self.run_per_image_standardize(data_points);

        return data_points, data_labels;

    def get_next_n(self, n:int):
        #print('self.last_fetched_indx = ', self.last_fetched_indx);
        ## validate parameters
        if(n <= 0):
            return None, None;

        ## make sure data is loaded first
        if(self.is_loaded == False):
            self.load_data();

        ## Get number of data points to retrieve        
        if(self.do_repeat == False):
            if (self.last_fetched_indx + n >= self.data_count):
                n = self.data_count - self.last_fetched_indx - 1;
                if(n <= 0):
                    return None, None;

        ## Get data shape
        data_size_x = self.input_img_width;
        data_size_y = self.input_img_height;    

        data_points, data_labels = self.get_next_one();

        #data_labels = np.zeros((n, self.n_classes))       
        #self.datapoints_files_list = [];# will be filled by load data point
        ##tic1 = time.time();
        #if(self.in_size_x is None):
        #    dp, data_labels[0,:] = self.get_next_one();
        #    self.in_size_y = dp.shape[0];
        #    self.in_size_x = dp.shape[1];
        #    data_points = np.zeros((n, self.in_size_y, self.in_size_x, self.n_channels))
        #    data_points[0,:,:,:] = dp;
        #    for i in range(1, n):
        #        #data_points[i], data_labels[i] = self.get_next_one();
        #        data_points[i, :,:,:], data_labels[i,:] = self.get_next_one();
        #        #datapoints_list.append(data_point_tmp);
        #        #labels_list.append(data_label_tmp);
        #else:
        #    data_points = np.zeros((n, self.in_size_y, self.in_size_x, self.n_channels))
        #    data_labels = np.zeros((n, self.n_classes))
        #    for i in range(0, n):
        #        #data_points[i], data_labels[i] = self.get_next_one();
        #        data_points[i, :,:,:], data_labels[i,:] = self.get_next_one();
        #        #datapoints_list.append(data_point_tmp);
        #        #labels_list.append(data_label_tmp);
  

        #if(self.do_augment == True):
        #    #tic1 = time.time();
        #    data_points, data_labels = self.augment_batch_new3(data_points, data_labels);
        #    ## debug
        #    #for i in range(0,n):
        #    #    io.imsave('/pylon5/ac3uump/shahira/tcga/tmp/'+ str(self.tmp_index) + '_' + str(i)+ '.png',  data_points[i,:,:,:].astype(np.int64));
        #    #tic2 = time.time();
        #    #print('time augment = ', tic2 - tic1);
        #    #sys.stdout.flush()

        #if(self.do_postprocess):
        #    tic1 = time.time();
        #    data_points, data_labels = self.postprocess_batch_new(data_points, data_labels);
        #    tic2 = time.time();
        #    print('time postprocess = ', tic2 - tic1);
        #    sys.stdout.flush()

        ## debug
        #if(not self.do_augment):
        #    for i in range(0,n):
        #        io.imsave('/pylon5/ac3uump/shahira/tcga/tmp/'+ str(self.tmp_index) + '_' + str(i)+ '.png',  data_points[i,:,:,:].astype(np.int64));

        #### standardize
        ##tic1 = time.time();
        ##data_points = self.run_per_image_standardize(data_points);
        #data_points = self.tf_data_points_std.eval(feed_dict={self.tf_data_points_std_in: data_points})
        ##tic2 = time.time();
        ##print('time standardize = ', tic2 - tic1);
        ##sys.stdout.flush()

        #### Norm with mean = 0 and std = 2
        np.clip(data_points, 0, 255, data_points);
        #print('data_points.shape = ', data_points.shape)
        #print('data_points.dtype = ', data_points.dtype)
        data_points = data_points.astype(np.float);
        data_points /= 255;
        data_points -= 0.5;
        data_points *= 2;


        return data_points, data_labels;

    def preprocess(self, data_point, label):
        data_point_list = [];
        for i in range(data_point.shape[0]):
            data_point2 = data_point[i];

            if(self.pre_crop_center):
                starty = (data_point2.shape[0] - self.pre_crop_height)//2;
                startx = (data_point2.shape[1] - self.pre_crop_width)//2;
                endy = starty + self.pre_crop_height;
                endx = startx + self.pre_crop_width;
                data_point2 = data_point2[starty:endy, startx:endx, :];
            if(not(data_point2.shape[0] == self.input_img_height) or not(data_point2.shape[1] == self.input_img_width)):
                if(self.pre_resize):
                    data_point2 = sktransform.resize(data_point2, (self.input_img_height, self.input_img_width), preserve_range=True, anti_aliasing=True).astype(np.uint8);
                    #data_point2 = tf.image.resize_images(data_point2, [self.input_img_height, self.input_img_width]);
                elif(self.pre_center):
                    diff_y = self.input_img_height - data_point2.shape[0];
                    diff_x = self.input_img_width - data_point2.shape[1];
                    diff_y_div2 = diff_y//2;
                    diff_x_div2 = diff_x//2;
                    data_point_tmp = np.zeros((self.input_img_height, self.input_img_width, data_point2.shape[2]));
                    if(diff_y >= 0 and diff_x >= 0):
                        data_point_tmp[diff_y:diff_y+self.input_img_height, diff_x:diff_x+self.input_img_width, :] = data_point2;
                        data_point2 = data_point_tmp;
                ##debug
                #print('after resize: ', data_point2.shape);
            data_point_list.append(data_point2);
        data_point2 = np.array(data_point_list);
        #print('after preprocess data_point2.shape = ', data_point2.shape)
        return data_point2, label;


    #def augment(self, data_point, label):
    #    '''
    #        select augmentation:        
    #        0: same (none)
    #        1: horizontal flip
    #        2: vertical flip
    #        3: horizontal and vertical flip
    #        4: rotate 90
    #        5: rotate 180
    #        6: rotate 270 or -90
    #    '''
    #    op = random.randint(0,7);
    #    data_point2 = data_point;
    #    label2 = label;
    #    if(op == 1):
    #        data_point2 = data_point[:,::-1,:];
    #        if(label is not None):
    #            label2 = label[:,::-1,:];
    #    elif(op == 2):
    #        data_point2 = data_point[::-1,:,:];
    #        if(label is not None):
    #            label2 = label[::-1,:,:];
    #    elif(op == 3):
    #        data_point2 = data_point[:,::-1,:];
    #        data_point2 = data_point2[::-1,:,:];
    #        if(label is not None):
    #            label2 = label[:,::-1,:];
    #            label2 = label2[::-1,:,:];
    #    elif(op == 4):
    #        data_point2 = np.rot90(data_point, k = 1, axes=(0,1));
    #        if(label is not None):
    #            label2 = np.rot90(label, k = 1, axes=(0,1));
    #    elif(op == 5):
    #        data_point2 = np.rot90(data_point, k=2, axes=(0,1));
    #        if(label is not None):
    #            label2 = np.rot90(label, k=2, axes=(0,1));
    #    elif(op == 6):
    #        data_point2 = np.rot90(data_point, k=3, axes=(0,1));
    #        if(label is not None):
    #            label2 = np.rot90(label, k=3, axes=(0,1));
    #    return data_point2, label2


    def load_datapoint(self, indx):
        filepath = os.path.join(self.filepath_data, self.data[indx]);
        
        #self.datapoints_files_list.append(filepath)
        #filename_queue = tf.train.string_input_producer([filepath]) #  list of files to read

        #reader = tf.WholeFileReader()
        #key, value = reader.read(filename_queue)

        #img = tf.image.decode_png(value) # use png or jpg decoder based on your files.
        #print('filepath=', filepath)
        img = io.imread(filepath);
        if(img.shape[2] > 3): # remove the alpha
            img = img[:,:,0:3];
        #label_str = filepath[-5];
        #if(label_str == '0'):
        #    #label = tf.convert_to_tensor([1, 0], dtype=tf.int8);
        #    label = np.array([1, 0], dtype=np.int8);
        #elif(label_str == '1'):
        #    #label = tf.convert_to_tensor([0, 1], dtype=tf.int8);
        #    label = np.array([0, 1], dtype=np.int8);
        #else:
        label = None;
        patch_height, patch_width = int(img.shape[0]//float(self.n_subpatches_per_side)), int(img.shape[1]//float(self.n_subpatches_per_side));
        #print('patch_height = ', patch_height)
        #print('patch_width = ', patch_width)
        data_point = np.zeros((int(self.n_subpatches_per_side*self.n_subpatches_per_side), int(patch_height), int(patch_width ), img.shape[2]));
        i = 0;
        for y in range(self.n_subpatches_per_side):
            for x in range(self.n_subpatches_per_side):
                data_point[i] =  img[y*patch_height : (y + 1) * patch_height, x*patch_width : (x + 1) * patch_width, :]
                #io.imsave('/pylon5/ac3uump/shahira/tcga/tmp/'+self.data[indx] + str(i) + '.png', data_point[i].astype(np.uint8));
                i += 1;
        #data_point = img;
            
        return data_point, label;

    # prepare the mapping from allowed operations to available operations index
    def create_augmentation_map(self, kwargs={}):
        args = {'aug_flip_h': 'True', 'aug_flip_v': 'True', 'aug_flip_hv': 'True' \
            , 'aug_rot180': 'True', 'aug_rot90': 'False', 'aug_rot270': 'False', 'aug_rot_rand': 'False' \
            , 'aug_brightness': 'False', 'aug_brightness_min': -50,  'aug_brightness_max': 50 \
            , 'aug_saturation': 'False', 'aug_saturation_min': -1.5,  'aug_saturation_max': 1.5 \
            , 'aug_hue': 'False', 'aug_hue_min': -50,  'aug_hue_max': 50 \
            , 'aug_scale': 'False', 'aug_scale_min': 1.0,  'aug_scale_max': 2.0 \
            , 'aug_translate': 'False',  'aug_translate_y_min': -20, 'aug_translate_y_max': 20,  'aug_translate_x_min': -20, 'aug_translate_x_max': 20
            };
        #print(args);
        args.update(kwargs);    
        #print(args);
        self.rot_angles = [];
        self.aug_flip_h = bool(strtobool(args['aug_flip_h']));
        self.aug_flip_v = bool(strtobool(args['aug_flip_v']));
        self.aug_flip_hv = bool(strtobool(args['aug_flip_hv']));
        self.aug_rot180 = bool(strtobool(args['aug_rot180']));
        self.aug_rot90 = bool(strtobool(args['aug_rot90']));
        self.aug_rot270 = bool(strtobool(args['aug_rot270']));
        self.aug_rot_random = bool(strtobool(args['aug_rot_rand']));
        self.aug_brightness = bool(strtobool(args['aug_brightness']));
        self.aug_saturation = bool(strtobool(args['aug_saturation']));
        self.aug_hue = bool(strtobool(args['aug_hue']));
        self.aug_scale = bool(strtobool(args['aug_scale']));
        self.aug_translate = bool(strtobool(args['aug_translate']));
        '''
        map allowed operation to the following values
        0: same (none)
        1: horizontal flip
        2: vertical flip
        3: horizontal and vertical flip
        4: rotate 180
        5: rotate 90
        6: rotate 270 or -90
        7: rotate random angle
        '''
        self.aug_map = {};
        self.aug_map[0] = 0; # (same) none 
        i = 1;
        if(self.aug_flip_h):
            self.aug_map[i] = 1;
            i += 1;
        if(self.aug_flip_v):
            self.aug_map[i] = 2;
            i += 1;
        if(self.aug_flip_hv):
            self.aug_map[i] = 3;
            i += 1;
        if(self.aug_rot180):
            self.aug_map[i] = 4;
            self.rot_angles.append(math.pi);
            i += 1;
        if(self.aug_rot90):
            #print('self.aug_rot90={}'.format(self.aug_rot90));
            self.aug_map[i] = 5;
            self.rot_angles.append(math.pi/2);
            i += 1;
        if(self.aug_rot270):
            #print('self.aug_rot270={}'.format(self.aug_rot270));
            self.aug_map[i] = 6;
            self.rot_angles.append(-math.pi/2);
            i += 1;
        if(self.aug_rot_random):
            #self.aug_map[i] = 7;
            self.aug_rot_min = int(args['aug_rot_min']);
            self.aug_rot_max = int(args['aug_rot_max']);
            for r in range(self.aug_rot_min, self.aug_rot_max+1, 5):
                self.rot_angles.append(r*math.pi/180);

        if(self.aug_brightness):
        #    self.aug_map[i] = 7;
            self.aug_brightness_min = int(args['aug_brightness_min']);
            self.aug_brightness_max = int(args['aug_brightness_max']);
            #print('self.aug_brightness_max=',self.aug_brightness_max);
            sys.stdout.flush();
        #    i += 1;
        if(self.aug_saturation):
            self.aug_saturation_min = float(args['aug_saturation_min']);
            self.aug_saturation_max = float(args['aug_saturation_max']);
        if(self.aug_hue):
            self.aug_hue_min = int(args['aug_hue_min']);
            self.aug_hue_max = int(args['aug_hue_max']);
        if(self.aug_scale):
            self.aug_scale_min = float(args['aug_scale_min']);
            self.aug_scale_max = float(args['aug_scale_max']);
        if(self.aug_translate):
            self.aug_translate_y_min = int(args['aug_translate_y_min']);
            self.aug_translate_y_max = int(args['aug_translate_y_max']);
            self.aug_translate_x_min = int(args['aug_translate_x_min']);
            self.aug_translate_x_max = int(args['aug_translate_x_max']);
        #print(self.aug_map)


    def augment(self, data_point, label):
        '''
            select augmentation:        
            0: same (none)
            1: horizontal flip
            2: vertical flip
            3: horizontal and vertical flip
            4: rotate 180
            5: rotate 90
            6: rotate 270 or -90
            7: rotate random
        '''        
        # because width and height are not equal cannot do rotation 90 and 270
        #op = random.randint(0,7);
        #print('data_point.shape');
        #print(data_point.shape);
        #op = random.randint(0,5);

        # select one of the valid operations and map it to its index in the available operations 
        op = random.randint(0,len(self.aug_map));
        op = self.aug_map[op];
        ## debug
        #print('op = ', op);
        data_point2 = data_point;
        # because this is a classification task, labels stay the same
        #label2 = label;
        # important: use ndarray.copy() when indexing with negative 
        #            It will alocate new memory for numpy array which make it normal, I mean the stride is not negative any more.
        #            Otherwise get the error: ValueError: some of the strides of a given numpy array are negative. This is currently not supported, but will be added in future releases.
        if(op == 1):
            #data_point2 = data_point[:,::-1,:].copy();
            data_point2 = tf.image.flip_left_right(data_point);
            #if(label is not None):
            #    label2 = label[:,::-1,:];
        elif(op == 2):
            #data_point2 = data_point[::-1,:,:].copy();
            data_point2 = tf.image.flip_up_down(data_point);
            #if(label is not None):
            #    label2 = label[::-1,:,:];
        elif(op == 3):
            #data_point2 = data_point[:,::-1,:];
            #data_point2 = data_point2[::-1,:,:].copy();
            data_point2 = tf.image.transpose_image(data_point);
            #if(label is not None):
            #    label2 = label[:,::-1,:];
            #    label2 = label2[::-1,:,:];
        elif(op == 4):
            #data_point2 = np.rot90(data_point, k=2, axes=(0,1)).copy();
            data_point2 = tf.image.rot90(data_point, k=2);
            #if(label is not None):
            #    label2 = np.rot90(label, k=2, axes=(0,1));
        elif(op == 5):
            #data_point2 = np.rot90(data_point, k=1, axes=(0,1)).copy();
            data_point2 = tf.image.rot90(data_point, k=1);
        #    if(label is not None):
        #        label2 = np.rot90(label, k = 1, axes=(0,1));
        elif(op == 6):
            #data_point2 = np.rot90(data_point, k=3, axes=(0,1)).copy();
            data_point2 = tf.image.rot90(data_point, k=3);
        #    if(label is not None):
        #        label2 = np.rot90(label, k=3, axes=(0,1));
        #elif(op == 7):
        if(self.aug_rot_random):
            angle = random.randint(self.aug_rot_min, self.aug_rot_max) * np.pi / 360.0;
            ##debug
            #print('angle = ', angle);
            #data_point2 = sktransform.rotate(data_point2, angle, preserve_range=True).astype(np.uint8);
            data_point2 = tf.contrib.image.rotate(data_point2, angle).astype(np.uint8);

        #    if(label is not None):
        #        label2 = np.rot90(label, k=3, axes=(0,1));

        ###debug
        #print('self.tmp_index = ', self.tmp_index);
        #print('self.img_id = ', self.img_id);

        op_brightness = random.random();
        op_saturation = random.random();
        op_hue = random.random();
        op_scale = random.random();
        if((self.aug_saturation and op_saturation > 0.5) or (self.aug_hue and op_hue > 0.5) or (self.aug_brightness and op_brightness > 0.5)):
            if(self.aug_hue and op_hue > 0.5):
                hue = random.randint(self.aug_hue_min, self.aug_hue_max);
                data_point2_hsv[:,:,0] += hue;
                data_point2_hsv[:,:,0][np.where(data_point2_hsv[:,:,0] > 179)] = 179;
                data_point2 = tf.image.random_hue(data_point2, max_delta=self.aug_hue_max/255.0)  # max_delta must be in the interval [0, 0.5]
            if(self.aug_saturation and op_saturation > 0.5):
                data_point2 = tf.image.random_saturation(data_point2, lower=self.aug_saturation_min, upper=self.aug_saturation_max);
            if(self.aug_brightness and op_brightness > 0.5):
                data_point2 = tf.image.random_brightness(data_point2, max_delta=self.aug_brightness_max/255.0)
            ##debug
            #print('hu-sat-br = ', hue, ' - ', saturation, ' - ', brightness);

            # The ranges that OpenCV manage for HSV format are the following:
            # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. Different softwares use different scales.
            #data_point2_hsv[np.where(data_point2_hsv < 0)] = 0;
            #data_point2_hsv = data_point2_hsv.astype(np.uint8);
            #data_point2 = cv.cvtColor(data_point2_hsv, cv.COLOR_HLS2RGB);
        #elif(self.aug_saturation and op_saturation > 0.5):
        #    #saturation = random.randint(self.aug_saturation_min, self.aug_saturation_max);
        #    saturation = random.random() * (self.aug_saturation_max-self.aug_saturation_min) + self.aug_saturation_min;
        #    ##debug
        #    print('sat= ', saturation)
        #    data_point2 = data_point2.astype(np.uint8);
        #    data_point2_hsv = cv.cvtColor(data_point2, cv.COLOR_RGB2HSV);
        #    data_point2_hsv = data_point2_hsv.astype(np.float)
        #    data_point2_hsv[:,:,1] *= saturation;
        #    data_point2_hsv[:,:,1][np.where(data_point2_hsv[:,:,1] > 255)] = 255;
        #    data_point2_hsv[np.where(data_point2_hsv < 0)] = 0;
        #    data_point2_hsv = data_point2_hsv.astype(np.uint8);
        #    data_point2 = cv.cvtColor(data_point2_hsv, cv.COLOR_HSV2RGB);
        #elif(self.aug_hue and op_hue > 0.5):
        #    #hue = random.random()*(self.aug_hue_max-self.aug_hue_min) + self.aug_hue_min;
        #    hue = random.randint(self.aug_hue_min, self.aug_hue_max);
        #    ##debug
        #    print('hu = ', hue)
        #    data_point2 = data_point2.astype(np.uint8);
        #    data_point2_hsv = cv.cvtColor(data_point2, cv.COLOR_RGB2HSV);
        #    data_point2_hsv = data_point2_hsv.astype(np.float)
        #    data_point2_hsv[:,:,0] += hue;
        #    data_point2_hsv[:,:,0][np.where(data_point2_hsv[:,:,0] > 179)] = 179;
        #    data_point2_hsv[np.where(data_point2_hsv < 0)] = 0;
        #    data_point2_hsv = data_point2_hsv.astype(np.uint8);
        #    data_point2 = cv.cvtColor(data_point2_hsv, cv.COLOR_HSV2RGB);
        #if(self.aug_brightness and op_brightness > 0.5):
        #    brightness = random.randint(self.aug_brightness_min, self.aug_brightness_max);
        #    ##debug
        #    print('br = ', brightness)
        #    data_point2 = data_point2.astype(np.int16)
        #    data_point2 += brightness;
        #    data_point2[np.where(data_point2 > 255)] = 255;
        #    data_point2[np.where(data_point2 < 0)] = 0;
        #    data_point2 = data_point2.astype(np.uint8);
        if(self.aug_translate):
            #translate_y = random.random()*(self.aug_translate_y_max-self.aug_translate_y_min) + self.aug_translate_y_min;
            #translate_x = random.random()*(self.aug_translate_x_max-self.aug_translate_x_min) + self.aug_translate_x_min;
            #translate_y = np.random.randint(self.aug_translate_y_min, high = self.aug_translate_y_max);
            #translate_x = np.random.randint(self.aug_translate_x_min, high = self.aug_translate_x_max);
            #translate_transform = sktransform.AffineTransform(translation = (translate_x, translate_y));      
            #data_point2 = sktransform.warp(data_point2, translate_transform, preserve_range=True).astype(np.uint8); 
            data_point2 = tf.py_func(self.op_translate, [data_point2], [tf.uint8]); 
        #if(self.aug_scale and op_scale > 0.5):
        #    scale = random.random()*(self.aug_scale_max-self.aug_scale_min) + self.aug_scale_min;
        #    ###debug
        #    #print('sc = ', scale)
        #    data_point2 = sktransform.rescale(data_point2, scale, preserve_range=True).astype(np.uint8);        
        #    scale_height, scale_width,_ = data_point2.shape;
        #    diff_height = scale_height - self.input_img_height;
        #    diff_width = scale_width - self.input_img_width;
        #    start_y = 0;
        #    start_x = 0;
        #    if(diff_height > 0):
        #        start_y = random.randint(0, diff_height);
        #    if(diff_width > 0):
        #        start_x = random.randint(0, diff_width);
        #    ####debug
        #    #io.imsave('/gpfs/projects/KurcGroup/sabousamra/debug/'+ str(self.tmp_index) + '_' + str(self.img_id)+ '_rescale.png',  data_point2);
        #    data_point2 = data_point2[start_y : start_y+self.input_img_height, start_x : start_x+self.input_img_width, : ]
        ##return data_point2, label2
        #print('data_point2.shape');
        #print(data_point2.shape);

        ###debug
        #print('self.tmp_index = ', self.tmp_index);
        #print('self.img_id = ', self.img_id);
        #print('label = ', label);
        #io.imsave('/gpfs/projects/KurcGroup/sabousamra/debug/'+ str(self.tmp_index) + '_' + str(self.img_id)+ '.png',  data_point2);
        self.tmp_index += 1;

        return tf.cast(tf.clip_by_value(tf.cast(data_point2, tf.int32), 0, 255), tf.uint8), label;

    def augment_batch(self, data_points, labels):
        data_points2 = data_points;
        count = data_points.shape[0];
        if(self.aug_flip_h or self.aug_flip_v):
            data_points2 = tf.image.random_flip_left_right(data_points2);
        if(self.aug_flip_v):
            data_points2 = tf.image.random_flip_up_down(data_points2);
        if(len(self.rot_angles) > 0):
            angles = np.random.choice(self.rot_angles, count);
            data_points2 = tf.contrib.image.rotate(data_points2, angles);
        #elif(op == 3):
        #    data_point2 = tf.image.transpose_image(data_point);
        ###debug
        #print('self.tmp_index = ', self.tmp_index);
        #print('self.img_id = ', self.img_id);
        if(self.aug_brightness):
            data_points2 = tf.image.random_brightness(data_points2, float(self.aug_brightness_max));
        if(self.aug_saturation):
            data_points2 = tf.image.random_saturation(data_points2, lower=self.aug_saturation_min, upper=self.aug_saturation_max);
        if(self.aug_hue):
            data_points2 = tf.image.random_hue(data_points2, self.aug_hue_max/255.0);
        if(self.aug_translate):
            translations_y = np.random.randint(self.aug_translate_y_min, high=self.aug_translate_y_max+1, size=count);
            translations_x = np.random.randint(self.aug_translate_x_min, high=self.aug_translate_x_max+1, size=count);
            translations = np.concatenate((translations_x.reshape((count,1)), translations_y.reshape((count,1))), axis=1);
            data_points2 = tf.contrib.image.translate(data_points2, translations);

  
        self.tmp_index += 1;

        return tf.cast(tf.clip_by_value(tf.cast(data_points2, tf.int32), 0, 255), tf.uint8), labels;

    def augment_batch_new(self, data_points, labels):
        data_points2 = data_points;
        count = data_points.shape[0];
        if(self.aug_flip_h or self.aug_flip_v):
            #data_points2 = tf.image.random_flip_left_right(data_points2);
            data_points2 = self.aug_flip(data_points2, labels)
        #if(self.aug_flip_v):
        #    data_points2 = tf.image.random_flip_up_down(data_points2);
        if(len(self.rot_angles) > 0):
            angles = np.random.choice(self.rot_angles, count);
            #data_points2 = tf.contrib.image.rotate(data_points2, angles);
            data_points2 = self.aug_rot(data_points2, labels, angles )
        #elif(op == 3):
        #    data_point2 = tf.image.transpose_image(data_point);
        ###debug
        #print('self.tmp_index = ', self.tmp_index);
        #print('self.img_id = ', self.img_id);
        if(self.aug_brightness):
            #data_points2 = tf.image.random_brightness(data_points2, float(self.aug_brightness_max));
            data_points2 = self.run_aug_brightness(data_points2);
        if(self.aug_saturation):
            #data_points2 = tf.image.random_saturation(data_points2, lower=self.aug_saturation_min, upper=self.aug_saturation_max);
            data_points2 = self.run_aug_saturation(data_points2);
        if(self.aug_hue):
            #data_points2 = tf.image.random_hue(data_points2, self.aug_hue_max/255.0);
            data_points2 = self.run_aug_hue(data_points2);
        if(self.aug_translate):
            translations_y = np.random.randint(self.aug_translate_y_min, high=self.aug_translate_y_max+1, size=count);
            translations_x = np.random.randint(self.aug_translate_x_min, high=self.aug_translate_x_max+1, size=count);
            translations = np.concatenate((translations_x.reshape((count,1)), translations_y.reshape((count,1))), axis=1);
            data_points2 = self.run_aug_translation(data_points2, translations);

  
        self.tmp_index += 1;

        #return tf.cast(tf.clip_by_value(tf.cast(data_points2, tf.int32), 0, 255), tf.uint8), labels;
        return data_points2, labels;

    def augment_batch_new2(self, data_points, labels):
        data_points2 = data_points;
        count = data_points.shape[0];

        angles = np.random.choice(self.rot_angles, count);
        translations_y = np.random.randint(self.aug_translate_y_min, high=self.aug_translate_y_max+1, size=count);
        translations_x = np.random.randint(self.aug_translate_x_min, high=self.aug_translate_x_max+1, size=count);
        translations = np.concatenate((translations_x.reshape((count,1)), translations_y.reshape((count,1))), axis=1);
        if(self.post_crop_y1 is None):
            self.post_crop_y1 = (data_points.shape[1] - self.post_crop_height)//2;
        if(self.post_crop_x1 is None):
            self.post_crop_x1 = (data_points.shape[2] - self.post_crop_width)//2;

        data_points2 = self.run_aug(data_points2, translations, angles);

  

        #return tf.cast(tf.clip_by_value(tf.cast(data_points2, tf.int32), 0, 255), tf.uint8), labels;
        return data_points2, labels;

    def augment_batch_new3(self, data_points, labels):
        data_points2 = data_points;
        count = data_points.shape[0];

        angles = np.random.choice(self.rot_angles, count);
        translations_y = np.random.randint(self.aug_translate_y_min, high=self.aug_translate_y_max+1, size=count);
        translations_x = np.random.randint(self.aug_translate_x_min, high=self.aug_translate_x_max+1, size=count);
        translations = np.concatenate((translations_x.reshape((count,1)), translations_y.reshape((count,1))), axis=1);
        if(self.post_crop_y1 is None):
            self.post_crop_y1 = (data_points.shape[1] - self.post_crop_height)//2;
        if(self.post_crop_x1 is None):
            self.post_crop_x1 = (data_points.shape[2] - self.post_crop_width)//2;

        data_points2 = self.tf_data_points_aug.eval(feed_dict={self.tf_data_points: data_points, self.tf_translations:translations, self.tf_angles:angles   
            , self.tf_post_crop_y1:self.post_crop_y1, self.tf_post_crop_x1:self.post_crop_x1
            })


        #return tf.cast(tf.clip_by_value(tf.cast(data_points2, tf.int32), 0, 255), tf.uint8), labels;
        return data_points2, labels;

    def op_translate(self, input):
        #print('translate - input.shape = ', input.shape);
        translate_y = np.random.randint(self.aug_translate_y_min, high = self.aug_translate_y_max);
        translate_x = np.random.randint(self.aug_translate_x_min, high = self.aug_translate_x_max);
        #print('translate - x,y = ', translate_x, translate_y);
        translate_transform = sktransform.AffineTransform(translation = (translate_x, translate_y));      
        data_point2 = sktransform.warp(input, translate_transform, preserve_range=True).astype(np.uint8); 
        #print('translate - data_point2.shape = ', data_point2.shape);
        return data_point2 ;

    def read_postprocess_parameters(self, kwargs={}):
        args = {'post_resize': 'False', 'post_crop_center': 'False',
            'post_crop_height': 128, 'post_crop_width': 128
            };
        #print(args);
        args.update(kwargs);    
        #print(args);
        self.post_resize = bool(strtobool(args['post_resize']));
        self.post_crop_center = bool(strtobool(args['post_crop_center']));
        if(self.post_crop_center ):
            self.post_crop_height = int(args['post_crop_height']);
            self.post_crop_width = int(args['post_crop_width']);
            self.post_crop_y1 = None;
            self.post_crop_x1 = None;

    def read_preprocess_parameters(self, kwargs={}):
        args = {'pre_resize': 'False', 'pre_crop_center': 'False',
            'pre_crop_height': 128, 'pre_crop_width': 128
            };
        #print(args);
        args.update(kwargs);    
        #print(args);
        self.pre_resize = bool(strtobool(args['pre_resize']));
        self.pre_crop_center = bool(strtobool(args['pre_crop_center']));
        if(self.pre_crop_center ):
            self.pre_crop_height = int(args['pre_crop_height']);
            self.pre_crop_width = int(args['pre_crop_width']);
            self.pre_crop_y1 = None;
            self.pre_crop_x1 = None;

    def postprocess(self, data_point, label):
        data_point2 = data_point.copy();
        if(data_point2.shape[0] == 1):
            data_point2 = data_point2.reshape((data_point2.shape[1], data_point2.shape[2], data_point2.shape[3]));
        #print('crop .shape = ', data_point.shape);
        if(self.post_crop_center):
            starty = (data_point2.shape[0] - self.post_crop_height)//2;
            startx = (data_point2.shape[1] - self.post_crop_width)//2;
            endy = starty + self.post_crop_height;
            endx = startx + self.post_crop_width;
            #print('self.post_crop_height = ', self.post_crop_height);
            #print('self.post_crop_width = ', self.post_crop_width);
            #print('starty = ', starty);
            #print('endy = ', endy);
            #print('startx = ', startx);
            #print('endx = ', endx);
            if(starty < 0 or startx < 0): # in case rotated the width and height will have changed
                starty = (data_point2.shape[0] - self.post_crop_width)//2;
                startx = (data_point2.shape[1] - self.post_crop_height)//2;
                endy = starty + self.post_crop_height;
                endx = startx + self.post_crop_width;
            
            ##debug
            #print('data_point2.shape = ', data_point2.shape);
            #print('starty = ', starty);
            #print('startx = ', startx);
            data_point2 = data_point2[starty:endy, startx:endx, :];
            ##debug
            #print('data_point2.shape = ', data_point2.shape);


        #print('resize - data_point2.shape', data_point2.shape);
        if(self.post_resize):
            data_point2 = sktransform.resize(data_point2, (self.input_img_height, self.input_img_width), preserve_range=True, anti_aliasing=True).astype(np.uint8);

        return data_point2, label;


    def postprocess_batch(self, data_points, labels):
        #print('crop .shape = ', data_point.shape);
        if(self.post_crop_center):
            if(self.post_crop_y1 is None):
                self.post_crop_y1 = (data_points.shape[1] - self.post_crop_height)//2;
            if(self.post_crop_x1 is None):
                self.post_crop_x1 = (data_points.shape[2] - self.post_crop_width)//2;
            data_points = tf.image.crop_to_bounding_box(data_points, self.post_crop_y1, self.post_crop_x1, self.post_crop_height, self.post_crop_width);


        #print('resize - data_point2.shape', data_point2.shape);
        if(self.post_resize):
            data_points = tf.image.resize_images(data_points, (self.input_img_height, self.input_img_width));

        return data_points, labels;

    def postprocess_batch_new(self, data_points, labels):
        #print('crop .shape = ', data_point.shape);
        if(self.post_crop_center):
            if(self.post_crop_y1 is None):
                self.post_crop_y1 = (data_points.shape[1] - self.post_crop_height)//2;
            if(self.post_crop_x1 is None):
                self.post_crop_x1 = (data_points.shape[2] - self.post_crop_width)//2;
            data_points = self.run_post_crop_center(data_points);


        #print('resize - data_point2.shape', data_point2.shape);
        if(self.post_resize):
            data_points = self.run_resize(data_points);

        return data_points, labels;

    def aug_flip(self, data_points, labels):
        tf_data_points = tf.placeholder(dtype=data_points.dtype)

        tf_data_points_tmp = tf.image.random_flip_left_right(tf_data_points);
        tf_data_points2 = tf.image.random_flip_up_down(tf_data_points_tmp);

        data_points2 = tf_data_points2.eval(feed_dict={tf_data_points: data_points})
        return data_points2  


    def aug_rot(self, data_points, labels, angles):
        tf_data_points = tf.placeholder(dtype=data_points.dtype, shape=(None, None, None, None))
        tf_angles = tf.placeholder(dtype=tf.float32, shape=(None,))
        tf_data_points2 = tf.contrib.image.rotate(tf_data_points, tf_angles);

        data_points2 = tf_data_points2.eval(feed_dict={tf_data_points: data_points, tf_angles:angles })
        return data_points2  


    def run_aug_brightness(self, data_points):
        tf_data_points = tf.placeholder(dtype=data_points.dtype, shape=(None, None, None, None))
        tf_data_points2 = tf.image.random_brightness(tf_data_points, float(self.aug_brightness_max));

        data_points2 = tf_data_points2.eval(feed_dict={tf_data_points: data_points })
        return data_points2  

    def run_aug_saturation(self, data_points):
        tf_data_points = tf.placeholder(dtype=data_points.dtype, shape=(None, None, None, None))
        tf_data_points2 = tf.image.random_saturation(tf_data_points, lower=self.aug_saturation_min, upper=self.aug_saturation_max);

        data_points2 = tf_data_points2.eval(feed_dict={tf_data_points: data_points })
        return data_points2  


    def run_aug_hue(self, data_points):
        tf_data_points = tf.placeholder(dtype=data_points.dtype, shape=(None, None, None, None))
        tf_data_points2 = tf.image.random_hue(tf_data_points, self.aug_hue_max/255.0);

        data_points2 = tf_data_points2.eval(feed_dict={tf_data_points: data_points })
        return data_points2  

    def run_aug_translation(self, data_points, translations):
        tf_data_points = tf.placeholder(dtype=data_points.dtype, shape=(None, None, None, None))
        #tf_translations = tf.placeholder(dtype=translations.dtype, shape=(None, None))
        tf_translations = tf.placeholder(dtype=tf.float32, shape=(None, None))

        tf_data_points2 = tf.contrib.image.translate(tf_data_points, tf_translations);

        data_points2 = tf_data_points2.eval(feed_dict={tf_data_points: data_points, tf_translations:translations  })
        return data_points2  

    def run_post_crop_center(self, data_points):
        tf_data_points = tf.placeholder(dtype=data_points.dtype, shape=(None, None, None, None))

        tf_data_points2 = tf.image.crop_to_bounding_box(tf_data_points, self.post_crop_y1, self.post_crop_x1, self.post_crop_height, self.post_crop_width);

        data_points2 = tf_data_points2.eval(feed_dict={tf_data_points: data_points  })
        return data_points2  

    def run_resize(self, data_points):
        tf_data_points = tf.placeholder(dtype=data_points.dtype, shape=(None, None, None, None))

        tf_data_points2 = tf.image.resize_images(tf_data_points, (self.input_img_height, self.input_img_width));

        data_points2 = tf_data_points2.eval(feed_dict={tf_data_points: data_points  })
        return data_points2  

    def run_per_image_standardize(self, data_points):
        tf_data_points = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None))

        tf_data_points2 = tf.map_fn(lambda img:tf.image.per_image_standardization(img), tf_data_points)

        data_points2 = tf_data_points2.eval(feed_dict={tf_data_points: data_points  })
        return data_points2  

    def save_state(self, checkpoint_filepath):
        if(checkpoint_filepath is None):
            return;
        base_filename,_ = os.path.splitext(checkpoint_filepath);
        if(self.permutation is not None):
            filepath_perm = base_filename + '_perm.npy' ;
            self.permutation.dump(filepath_perm);
        if(self.data is not None):
            filepath_data = base_filename + '_dat.pkl' ;
            pickle.dump(self.data, open(filepath_data, 'wb'));
        filepath_param = base_filename + '_param.pkl' ;
        pickle.dump([self.last_fetched_indx, self.data_count], open(filepath_param, 'wb'));

    def restore_state(self, checkpoint_filepath):
        if(checkpoint_filepath is None):
            return;
        base_filename,_ = os.path.splitext(checkpoint_filepath);
        filepath_perm = base_filename + '_perm.npy' ;
        if(os.path.isfile(filepath_perm)):            
            self.permutation = np.load(filepath_perm);
        filepath_data = base_filename + '_dat.pkl' ;
        if(os.path.isfile(filepath_data)):            
            self.data = pickle.load(open(filepath_data, 'rb'));
        filepath_param = base_filename + '_param.pkl' ;
        if(os.path.isfile(filepath_param)):            
            self.last_fetched_indx, self.data_count = pickle.load(open(filepath_param, 'rb'));

        print('self.last_fetched_indx, self.data_count = ', self.last_fetched_indx, self.data_count)
        
    def run_aug(self, data_points, translations, angles):
        tf_data_points = tf.placeholder(dtype=data_points.dtype, shape=(None, None, None, None))
        tf_translations = tf.placeholder(dtype=tf.float32, shape=(None, None))
        tf_angles = tf.placeholder(dtype=tf.float32, shape=(None,))

        tf_data_points_tmp1 = tf.image.random_flip_left_right(tf_data_points);
        tf_data_points_tmp2 = tf.image.random_flip_up_down(tf_data_points_tmp1);
        tf_data_points_tmp3 = tf.contrib.image.rotate(tf_data_points_tmp2, tf_angles);
        tf_data_points_tmp4 = tf.image.random_brightness(tf_data_points_tmp3, float(self.aug_brightness_max));
        tf_data_points_tmp5 = tf.contrib.image.translate(tf_data_points_tmp4, tf_translations);
        tf_data_points_tmp6 = tf.image.crop_to_bounding_box(tf_data_points_tmp5, self.post_crop_y1, self.post_crop_x1, self.post_crop_height, self.post_crop_width);
        tf_data_points2 = tf.image.resize_images(tf_data_points_tmp6, (self.input_img_height, self.input_img_width));
                        

        data_points2 = tf_data_points2.eval(feed_dict={tf_data_points: data_points, tf_translations:translations, tf_angles:angles   })


        return data_points2  

    def write_label_info(self, outpath, prefix):
        filepath = os.path.join(outpath, prefix + '_cancer_type.pkl');
        pickle.dump(self.cancer_type_list, open(filepath, 'wb'));

        filepath = os.path.join(outpath, prefix + '_filename.pkl');
        pickle.dump(self.filename_list, open(filepath, 'wb'));

        filepath = os.path.join(outpath, prefix + '_individual_labels.npy');
        np.array(self.individual_labels_list).dump(filepath)

        filepath = os.path.join(outpath, prefix + '_avg_label.npy');
        np.array(self.avg_label_list).dump(filepath);

        filepath = os.path.join(outpath, prefix + '_pred_old.npy');
        np.array(self.pred_old_list).dump(filepath);
