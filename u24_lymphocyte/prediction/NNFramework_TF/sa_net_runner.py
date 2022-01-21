# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import sys;
import os;
import tensorflow as tf;
import configparser;

#import net_model_params;
from sa_net_data_provider import AbstractDataProvider;
from sa_data_providers.mat_data_provider import MatDataProvider;
from sa_networks.unet_arch import UNetArch;
from sa_net_loss_func import CostFuncTypes;
from sa_net_train import CNNTrainer;
from sa_net_optimizer import OptimizerTypes;

if __name__ == "__main__":
    sys.path.append("..");

    # Read input arguments
    valid_arg_count = 1;
    arg_count = len(sys.argv);
    print('number of arguments = ' +  str(arg_count));
    for val in sys.argv:
        print(val);

    if(arg_count != valid_arg_count):
        print('error: number of arguments <> {}'.format(valid_arg_count));
        sys.exit();

    config_filepath = sys.argv[1];

    # read the config file
    config = configparser.ConfigParser();

    try:
        config.read(config_filepath);        
        network_class_name = config['NETWORK']['class_name'].strip();  ## strip: trims white space
        network_params = dict(config.items('NETWORK'));
        cost_class_name = config['COST']['class_name'].strip();  ## strip: trims white space
        cost_params = dict(config.items('COST'));
        train_class_name = config['TRAIN']['class_name'].strip();  ## strip: trims white space
        train_params = dict(config.items('TRAIN'));
    except:
        capFilepath_ini = '';
        outDir_ini = '';

    train_data_provider = MatDataProvider( \
        is_test=False \
        , filepath_data = 'C:\\Data\\ComputerVision\\Zebrafish\\used\\new2018-04-19\\unet_anno\\cap12-13-14-16_10_end_LCN4_avg.mat' \
        , filepath_label = 'C:\\Data\\ComputerVision\\Zebrafish\\used\\new2018-04-19\\unet_anno\\img_label_3classes_sqr_cap12-13-14-16_10_end.mat' \
        , n_channels = 1 \
        , n_classes = 3 \
        , do_preprocess = False \
        , do_augment = True \
        , data_var_name = 'c488gb' \
        , label_var_name = 'labelImg' \
        , permute = True \
        , repeat = True \
    );

    ########### to do: should have list of files
    test_data_provider = MatDataProvider( \
        is_test=True \
        , filepath_data = 'C:\\Data\\ComputerVision\\Zebrafish\\used\\new2018-04-19\\unet_anno\\cap19gb_LCN4_avg.mat' \
        , filepath_label = 'C:\\Data\\ComputerVision\\Zebrafish\\used\\new2018-04-19\\unet_anno\\img_label_3classes_sqr_cap19.mat' \
        , n_channels = 1 \
        , n_classes = 3 \
        , do_preprocess = False \
        , do_augment = False \
        , data_var_name= 'c488gb' \
        , label_var_name= 'labelImg' \
        , permute=True \
        , repeat = False \
    );

    input_x = tf.placeholder(tf.float32, [None, None, None, None])

    model_out_path = os.path.join('C:','Data','ComputerVision','NNFramework','test_out');
    print('model_out_path = '+ model_out_path);
    model_base_filename = 'unet_model';

    cost_func = CrossEntropyCost(n_classes = 3, kwargs=cost_params);

    cnn_arch = UNetArch(n_channels = 1, n_classes = 3, model_out_path = model_out_path, model_base_filename = model_base_filename, cost_func = cost_func \
        , kwargs=network_params \
        #, kwargs={'n_layers_per_path':3, 'n_conv_blocks_in_start': 64, 'block_size':3, 'pool_size':2 \
        #, 'cost_func': CostFuncTypes.CROSS_ENTROPY \
        #}
        );
    #cnn_arch.set_class_weights([1.0, 1.0, 1.0], kwargs={});

    # define session
    # restore model
    # unet_cnn.restore_model();

    #session_config = tf.ConfigProto(device_count = {'GPU': 1});
    #session_config.gpu_options.per_process_gpu_memory_fraction = 1.0;
    session_config = None;
    
    #train_args = {'max_epochs':1000, 'learning_rate': 0.0005, 'batch_size':1, 'epoch_size':10}

    trainer = CNNTrainer(cnn_arch \
        , train_data_provider \
        , optimizer_type=OptimizerTypes.ADAM \
        , session_config=session_config \
        , kwargs = train_params \
    );
    
    dropout = 0.0;
    if('dropout' in train_params):
        dropout = train_params['dropout'];
    trainer.train(do_init=True, do_restore=True, do_load_data=True, dropout=dropout, display_step=5);