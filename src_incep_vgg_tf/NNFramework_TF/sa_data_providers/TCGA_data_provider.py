# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
from ..sa_net_data_provider import AbstractDataProvider;
from numpy import random;

#import scipy.io as spio;
import numpy as np;
import glob;
import os;
import pickle;
from distutils.util import strtobool;
import tensorflow as tf;
from skimage import io;
from skimage import transform as sktransform;


class TCGADataProvider(AbstractDataProvider):
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
        if(do_augment):
            self.create_augmentation_map(kwargs);
        if(self.do_postprocess):
            self.read_postprocess_parameters(kwargs); 


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

        self.is_loaded = False;
        self.tmp_index = 0;

    def load_data(self):
        self.data = None;
        self.label = None;
        self.last_fetched_indx = -1;
        self.permutation = None;
        self.data_count = 0;
        self.data = None;
        self.labels = None;

        file_pattern = '*.png';
        file_pattern_full = os.path.join(self.filepath_data, '**', file_pattern);
        #print('file_pattern_full = ', file_pattern_full );
        self.data = glob.glob(file_pattern_full, recursive=True);
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
        data_point, label, self.datapoint_filename = self.load_datapoint(actual_indx);

        ## process the data
        if(self.do_preprocess == True):
            data_point, label = self.preprocess(data_point, label);
       
        if(self.do_augment == True):
            data_point, label = self.augment(data_point, label);

        if(self.do_postprocess):
            #data_point, label = self.postprocess(data_point, label);
            data_point, label = tf.py_func(self.postprocess, [data_point, label], [tf.uint8, tf.int8]); 

        ## normalize [0,1]
        #data_point /= data_point.max();
        ## normalize [-1,1]        
        #data_point = tf.subtract(data_point, 0.5);
        #data_point = tf.multiply(data_point, 2.0);
        data_point = tf.image.per_image_standardization(data_point);

        if(self.filepath_label == None):
            return data_point;
        else:
            return data_point, label;

    ## returns None, None if there is no more data to retrieve and repeat = false
    def get_next_n(self, n:int):
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
        self.datapoints_files_list = [];
        if(self.filepath_label is None):
            data_labels = None;
            labels_list = None;
        else:
            labels_list = [];
        
    
        for i in range(0, n):
            if(labels_list is None):
                #data_points[i] = self.get_next_one();
                datapoints_list.append(self.get_next_one());
                self.datapoints_files_list.append(self.datapoint_filename)
            else:
                #data_points[i], data_labels[i] = self.get_next_one();
                data_point_tmp, data_label_tmp = self.get_next_one();
                datapoints_list.append(data_point_tmp);
                labels_list.append(data_label_tmp);
                self.datapoints_files_list.append(self.datapoint_filename)

        data_points = tf.stack(datapoints_list);
        if(labels_list is not None):
            data_labels = tf.stack(labels_list);
        #print('data_points =' + str(np.shape(data_points)))
    
        return data_points, data_labels;

    def preprocess(self, data_point, label):
        data_point2 = data_point;

        if(not(data_point.shape[0] == self.input_img_height) or not(data_point.shape[1] == self.input_img_width)):
            ##debug
            #print('before resize: ', data_point2.shape);
            if(self.pre_resize):
                #data_point2 = sktransform.resize(data_point, (self.input_img_height, self.input_img_width), preserve_range=True, anti_aliasing=True).astype(np.uint8);
                data_point2 = tf.image.resize_images(data_point2, [self.input_img_height, self.input_img_width]);
        #    elif(self.pre_center):
        #        diff_y = self.input_img_height - data_point.shape[0];
        #        diff_x = self.input_img_width - data_point.shape[1];
        #        diff_y_div2 = diff_y//2;
        #        diff_x_div2 = diff_x//2;
        #        data_point2 = np.zeros((self.input_img_height, self.input_img_width, data_point.shape[2]));
        #        if(diff_y >= 0 and diff_x >= 0):
        #            data_point2[diff_y:diff_y+self.input_img_height, diff_x:diff_x+self.input_img_width, :] = data_point;
            ##debug
            #print('after resize: ', data_point2.shape);
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
        filepath = self.data[indx];
        img = io.imread(filepath);
        if(img.shape[2] > 3): # remove the alpha
            img = img[:,:,0:3];
        label_str = filepath[-5];
        if(label_str == '0'):
            label = tf.convert_to_tensor([1, 0], dtype=tf.int8);
        elif(label_str == '1'):
            label = tf.convert_to_tensor([0, 1], dtype=tf.int8);
        else:
            label = None;
        data_point = tf.convert_to_tensor(img);
        return data_point, label, os.path.split(filepath)[1];

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
            i += 1;
        if(self.aug_rot90):
            #print('self.aug_rot90={}'.format(self.aug_rot90));
            self.aug_map[i] = 5;
            i += 1;
        if(self.aug_rot270):
            #print('self.aug_rot270={}'.format(self.aug_rot270));
            self.aug_map[i] = 6;
            i += 1;
        if(self.aug_rot_random):
            #self.aug_map[i] = 7;
            self.aug_rot_min = int(args['aug_rot_min']);
            self.aug_rot_max = int(args['aug_rot_max']);
        if(self.aug_brightness):
        #    self.aug_map[i] = 7;
            self.aug_brightness_min = int(args['aug_brightness_min']);
            self.aug_brightness_max = int(args['aug_brightness_max']);
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
