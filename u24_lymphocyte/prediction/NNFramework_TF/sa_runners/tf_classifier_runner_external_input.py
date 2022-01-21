import sys;
import os;
import tensorflow as tf;
import configparser;

def load_model(config_filepath):
    #sys.path.append("..");
    #sys.path.append(".");


    device_ids_str = None;

    # read the gpu ids to use from the command line parameters if cuda is available
    if(not (device_ids_str is None)):
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str;

    # read the config file
    config = configparser.ConfigParser();

    config.read(config_filepath);  
    # General config
    config_name = config['DEFAULT']['config_name'].strip();  ## strip: trims white space
    running_mode = config['DEFAULT']['mode'].strip();  
    if(running_mode == 'train'):
        is_test = False;
    else:
        is_test = True;
    model_path = config['DEFAULT']['model_path'].strip();
    print('model_path = '+ model_path);
    model_base_filename = config['DEFAULT']['model_base_filename'].strip();
    if('model_restore_filename' in config['DEFAULT']):
        model_restore_filename = config['DEFAULT']['model_restore_filename'].strip();
    else:
        model_restore_filename = None;
    #model_fullpath = os.path.join(model_path, model_filename);

    # Network config
    network_params = dict(config.items('NETWORK'));
    network_class_name = config['NETWORK']['class_name'].strip();  
    n_channels = int(config['NETWORK']['n_channels'].strip());  
    n_classes = int(config['NETWORK']['n_classes'].strip());  

    # Cost config
    cost_params = dict(config.items('COST'));
    cost_func_class_name = config['COST']['class_name'].strip();  

    has_validation = False;
    if(is_test == False):
        # Train Data config
        train_params = dict(config.items('TRAIN_DATA'));
        train_config = config['TRAIN_DATA'];
        train_dataprovider_class_name = train_config['provider_class_name'].strip();  
        train_filepath_data = train_config['filepath_data'].strip();  
        train_filepath_label = None;
        if('filepath_label' in train_config):
            train_filepath_label = train_config['filepath_label'].strip();          
        train_preprocess = train_config.getboolean('preprocess');
        train_augment = train_config.getboolean('augment');
        train_permute = train_config.getboolean('permute');

        # Trainer config
        trainer_params = dict(config.items('TRAINER'));
        trainer_config = config['TRAINER'];
        trainer_class_name = trainer_config['class_name'].strip();  
        trainer_optimizer_type = trainer_config['optimizer_type'].strip();  

        # Validation Data config
        if('VALIDATE_DATA' in config):
            has_validation = True;
            validate_params = dict(config.items('VALIDATE_DATA'));
            validate_config = config['VALIDATE_DATA'];
            validate_dataprovider_class_name = validate_config['provider_class_name'].strip();  
            validate_filepath_data = validate_config['filepath_data'].strip();  
            validate_filepath_label = None;
            if('filepath_label' in validate_config):
                validate_filepath_label = validate_config['filepath_label'].strip();          
            validate_preprocess = validate_config.getboolean('preprocess');
            validate_augment = validate_config.getboolean('augment');
            validate_permute = validate_config.getboolean('permute');

    else:
        # Test Data config
        test_params = dict(config.items('TEST_DATA'));
        test_config = config['TEST_DATA'];
        test_dataprovider_class_name = test_config['provider_class_name'].strip();  
        test_filepath_data = test_config['filepath_data'].strip();  
        test_filepath_label = None;
        if('filepath_label' in test_config):
            test_filepath_label = test_config['filepath_label'].strip();          
        test_preprocess = test_config.getboolean('preprocess');
        test_augment = test_config.getboolean('augment');
        #test_permute = test_config.getboolean('permute');

        # Tester config
        tester_params = dict(config.items('TESTER'));
        tester_config = config['TESTER'];
        tester_class_name = tester_config['class_name'].strip();  
        tester_out_dir = tester_config['out_dir'].strip();  
        tester_out_ext = tester_config['out_ext'].strip();  


    if(cost_func_class_name == 'MSECost'):
        from ..sa_cost_func.mse_cost_func import MSECost;
        cost_func = MSECost(n_classes = n_classes, kwargs=cost_params);
    elif(cost_func_class_name == 'CrossEntropyCost'):
        from ..sa_cost_func.cross_entropy_cost_func import CrossEntropyCost;
        cost_func = CrossEntropyCost(n_classes = n_classes, kwargs=cost_params);
    else:
        print('error: cost function class name \'{}\' is not supported by runner'.format(cost_func_class_name));
        sys.exit();        

    
    if(network_class_name == 'SimpleClassifier'):
        cnn_arch = SimpleClassifierArch(n_channels = n_channels, n_classes = n_classes, model_out_path = model_path, model_base_filename = model_base_filename, model_restore_filename = model_restore_filename, cost_func = cost_func \
            , kwargs=network_params \
            );
    elif(network_class_name == 'InceptionResnetV2Classifier'):
        from ..sa_networks.inception_resnet_v2_classifier_arch import InceptionResnetV2ClassifierArch;
        cnn_arch = InceptionResnetV2ClassifierArch(n_channels = n_channels, n_classes = n_classes, model_out_path = model_path, model_base_filename = model_base_filename, model_restore_filename = model_restore_filename, cost_func = cost_func \
            , kwargs=network_params \
            );
    elif(network_class_name == 'InceptionV4Classifier'):
        from ..sa_networks.inception_v4_classifier_arch import InceptionV4ClassifierArch;
        cnn_arch = InceptionV4ClassifierArch(n_channels = n_channels, n_classes = n_classes, model_out_path = model_path, model_base_filename = model_base_filename, model_restore_filename = model_restore_filename, cost_func = cost_func \
            , kwargs=network_params \
            );
    elif(network_class_name == 'Resnet101Classifier'):
        from ..sa_networks.resnet_101_classifier_arch import Resnet101ClassifierArch;
        cnn_arch = Resnet101ClassifierArch(n_channels = n_channels, n_classes = n_classes, model_out_path = model_path, model_base_filename = model_base_filename, model_restore_filename = model_restore_filename, cost_func = cost_func \
            , kwargs=network_params \
            );
    elif(network_class_name == 'Resnet152Classifier'):
        from ..sa_networks.resnet_152_classifier_arch import Resnet152ClassifierArch;
        cnn_arch = Resnet152ClassifierArch(n_channels = n_channels, n_classes = n_classes, model_out_path = model_path, model_base_filename = model_base_filename, model_restore_filename = model_restore_filename, cost_func = cost_func \
            , kwargs=network_params \
            );
    elif(network_class_name == 'Resnet50Classifier'):
        from ..sa_networks.resnet_50_classifier_arch import Resnet50ClassifierArch;
        cnn_arch = Resnet50ClassifierArch(n_channels = n_channels, n_classes = n_classes, model_out_path = model_path, model_base_filename = model_base_filename, model_restore_filename = model_restore_filename, cost_func = cost_func \
            , kwargs=network_params \
            );
    elif(network_class_name == 'Resnet18Classifier'):
        from ..sa_networks.resnet_18_classifier_arch import Resnet18ClassifierArch;
        cnn_arch = Resnet18ClassifierArch(n_channels = n_channels, n_classes = n_classes, model_out_path = model_path, model_base_filename = model_base_filename, model_restore_filename = model_restore_filename, cost_func = cost_func \
            , kwargs=network_params \
            );
    elif(network_class_name == 'VGG16Classifier'):
        from ..sa_networks.vgg_16_classifier_arch import VGG16ClassifierArch;
        cnn_arch = VGG16ClassifierArch(n_channels = n_channels, n_classes = n_classes, model_out_path = model_path, model_base_filename = model_base_filename, model_restore_filename = model_restore_filename, cost_func = cost_func \
            , kwargs=network_params \
            );
    else:
        print('error: network class name \'{}\' is not supported by runner'.format(network_class_name));
        sys.exit();  

    if(is_test == False):
        if(train_dataprovider_class_name == 'TCGADataProvider'):
            from ..sa_data_providers.TCGA_data_provider import TCGADataProvider;
            train_data_provider = TCGADataProvider( \
                is_test=is_test \
                , filepath_data = train_filepath_data \
                , filepath_label = train_filepath_label \
                , n_channels = n_channels \
                , n_classes = n_classes \
                , do_preprocess = train_preprocess \
                , do_augment = train_augment \
                , data_var_name = None \
                , label_var_name = None \
                , permute = train_permute \
                , repeat = True \
                , kwargs = train_params\
            );
        elif(train_dataprovider_class_name == 'TCGABatchDataProvider'):
            from ..sa_data_providers.TCGA_batch_data_provider import TCGABatchDataProvider;
            train_data_provider = TCGABatchDataProvider( \
                is_test=is_test \
                , filepath_data = train_filepath_data \
                , filepath_label = train_filepath_label \
                , n_channels = n_channels \
                , n_classes = n_classes \
                , do_preprocess = train_preprocess \
                , do_augment = train_augment \
                , data_var_name = None \
                , label_var_name = None \
                , permute = train_permute \
                , repeat = True \
                , kwargs = train_params\
            );
        elif(train_dataprovider_class_name == 'TCGABatchEqualDataProvider'):
            from ..sa_data_providers.TCGA_batch_equal_data_provider import TCGABatchEqualDataProvider;
            train_data_provider = TCGABatchEqualDataProvider( \
                is_test=is_test \
                , filepath_data = train_filepath_data \
                , filepath_label = train_filepath_label \
                , n_channels = n_channels \
                , n_classes = n_classes \
                , do_preprocess = train_preprocess \
                , do_augment = train_augment \
                , data_var_name = None \
                , label_var_name = None \
                , permute = train_permute \
                , repeat = True \
                , kwargs = train_params\
            );
        else:
            print('error: train data provider class name \'{}\' is not supported by runner'.format(train_dataprovider_class_name));
            sys.exit();       

        if(has_validation):
            if(validate_dataprovider_class_name == 'TCGADataProvider'):
                from ..sa_data_providers.TCGA_data_provider import TCGADataProvider;
                validate_data_provider = TCGADataProvider( \
                    is_test=is_test \
                    , filepath_data = validate_filepath_data \
                    , filepath_label = validate_filepath_label \
                    , n_channels = n_channels \
                    , n_classes = n_classes \
                    , do_preprocess = validate_preprocess \
                    , do_augment = validate_augment \
                    , data_var_name = None \
                    , label_var_name = None \
                    , permute = validate_permute \
                    , repeat = False \
                    , kwargs = validate_params\
                ); 
            elif(validate_dataprovider_class_name == 'TCGABatchDataProvider'):
                from ..sa_data_providers.TCGA_batch_data_provider import TCGABatchDataProvider;
                validate_data_provider = TCGABatchDataProvider( \
                    is_test=is_test \
                    , filepath_data = validate_filepath_data \
                    , filepath_label = validate_filepath_label \
                    , n_channels = n_channels \
                    , n_classes = n_classes \
                    , do_preprocess = validate_preprocess \
                    , do_augment = validate_augment \
                    , data_var_name = None \
                    , label_var_name = None \
                    , permute = validate_permute \
                    , repeat = False \
                    , kwargs = validate_params\
                ); 
    else:
        if(test_dataprovider_class_name == 'TCGADataProvider'):
            from ..sa_data_providers.TCGA_data_provider import TCGADataProvider;
            ########### to do: should allow list of files
            test_data_provider = TCGADataProvider( \
                is_test=is_test \
                , filepath_data = test_filepath_data \
                , filepath_label = None \
                , n_channels = n_channels \
                , n_classes = n_classes \
                , do_preprocess = test_preprocess \
                , do_augment = test_augment \
                , data_var_name = None \
                , label_var_name = None \
                , permute = False \
                , repeat = False \
                , kwargs = test_params\
            );
        elif(test_dataprovider_class_name == 'TCGABatchDataProvider'):
            from ..sa_data_providers.TCGA_batch_data_provider import TCGABatchDataProvider;
            ########### to do: should allow list of files
            test_data_provider = TCGABatchDataProvider( \
                is_test=is_test \
                , filepath_data = test_filepath_data \
                , filepath_label = None \
                , n_channels = n_channels \
                , n_classes = n_classes \
                , do_preprocess = test_preprocess \
                , do_augment = test_augment \
                , data_var_name = None \
                , label_var_name = None \
                , permute = False \
                , repeat = False \
                , kwargs = test_params\
            );
        elif(not (test_dataprovider_class_name.lower() == 'none')):
            print('error: test data provider class name \'{}\' is not supported by runner'.format(test_dataprovider_class_name));
            sys.exit();        




    # define session
    # restore model
    # unet_cnn.restore_model();

    #session_config = tf.ConfigProto(device_count = {'GPU': 1});
    #session_config.gpu_options.per_process_gpu_memory_fraction = 1.0;
    session_config = None;
    
    #train_args = {'max_epochs':1000, 'learning_rate': 0.0005, 'batch_size':1, 'epoch_size':10}

    if(is_test == False):
        from ..sa_net_optimizer import OptimizerTypes;
        if(trainer_optimizer_type == 'ADAM'):
            optimizer_type=OptimizerTypes.ADAM;
        elif(trainer_optimizer_type == 'SGD'):
            optimizer_type=OptimizerTypes.SGD;
        else:
            print('error: trainer optimizer type \'{}\' is not supported by runner'.format(trainer_class_name));
            sys.exit();        
    
        if(trainer_class_name == 'ClassifierTrainer'):
            from ..sa_trainers.sa_net_train_classifier import ClassifierTrainer;
            trainer = ClassifierTrainer(cnn_arch \
                , train_data_provider \
                , validate_data_provider \
                , optimizer_type=optimizer_type \
                , session_config=session_config \
                , kwargs = trainer_params \
            );
        elif(trainer_class_name == 'ClassifierTrainerOpt'):
            from ..sa_trainers.sa_net_train_opt_classifier import ClassifierTrainerOpt;
            trainer = ClassifierTrainerOpt(cnn_arch \
                , train_data_provider \
                , validate_data_provider \
                , optimizer_type=optimizer_type \
                , session_config=session_config \
                , kwargs = trainer_params \
            );
        elif(trainer_class_name == 'ClassifierTrainerMultiGPU'):
            from ..sa_trainers.sa_net_train_multi_gpu_classifier import ClassifierTrainerMultiGPU;
            trainer = ClassifierTrainerMultiGPU(cnn_arch \
                , train_data_provider \
                , validate_data_provider \
                , optimizer_type=optimizer_type \
                , session_config=session_config \
                , kwargs = trainer_params \
            );
        else:
            print('error: trainer class name \'{}\' is not supported by runner'.format(trainer_class_name));
            sys.exit();        
    else:
        if(tester_class_name == 'ClassifierTester'):
            from ..sa_testers.sa_net_test_classifier import ClassifierTester;
            tester = ClassifierTester(cnn_arch \
                , test_data_provider \
                , session_config=session_config \
                , output_dir=tester_out_dir \
                , output_ext=tester_out_ext \
                , kwargs = tester_params \
            );
        elif(tester_class_name == 'ClassifierTesterExternalInput'):
            from ..sa_testers.sa_net_test_classifier_external_input import ClassifierTesterExternalInput;
            tester = ClassifierTesterExternalInput(cnn_arch \
                , session_config=session_config \
                , output_dir=tester_out_dir \
                , output_ext=tester_out_ext \
                , kwargs = tester_params \
            );
            tester.init_model(do_init=True, do_restore=True);
        elif(tester_class_name == 'ClassifierTesterExternalInputBinaryOutput'):
            from ..sa_testers.sa_net_test_classifier_external_input_binary_output import ClassifierTesterExternalInputBinaryOutput;
            tester = ClassifierTesterExternalInputBinaryOutput(cnn_arch \
                , session_config=session_config \
                , output_dir=tester_out_dir \
                , output_ext=tester_out_ext \
                , kwargs = tester_params \
            );
            tester.init_model(do_init=True, do_restore=True);
        else:
            print('error: tester class name \'{}\' is not supported by runner'.format(tester_class_name));
            sys.exit();        
    
    if(is_test == False):
        #trainer.train(do_init=True, do_restore=True, do_load_data=True);
        return trainer;
    else:
        #tester.test(do_init=True, do_restore=True, do_load_data=True);
        return tester;

