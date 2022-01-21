
## CNN training with Tensorflow

### Configuration file:  
To run training create a configuration file similar to config/config_incep-all.ini.  
To run test create a configuration file similar to one of the ini files under config/*_test/.  
Below are the main parameters that may need to be adjusted for a run:  

##### DEFAULT section:
**mode:** supported values (*train, test*)  
**model_path:** The output folder  
**model_base_filename:** All output files will have this prefix so it is preferably distinctive of the training configuration  
**model_restore_filename:** The full path of a pretrained model or a previous checkpoint (*optional*)  
For ImageNet pretrained models we used:  https://github.com/tensorflow/models/tree/master/research/slim

##### NETWORK section:
**class_name:** currently supported values are (*VGG16Classifier, Resnet101Classifier, Resnet152Classifier, InceptionV4Classifier, InceptionResnetV2Classifier*)  
**input_img_height, input_img_width:** the final image height and width that are fed into the network.  
**official_checkpoint:** supported values (*true, false*). If true, the final fully connected layer is excluded from restoration.  

##### TRAIN_DATA section:
**provider_class_name:** currently supported values are (*TCGADataProvider, TCGABatchDataProvider*)  
**filepath_data:** The path of the data files  
**filepath_label:** The path of the labels (if there are labels)   
**preprocess:** supported values (*true, false*). Preprocessing takes place before augmentation. If true then set either pre_resize or pre_center.  
**augment:** supported values (*true, false*). If true then setup the augmentation parameters (has the pattern *aug_**)  
**postprocess:** supported values (*true, false*). If true then setup the post processing parameters (*(post_crop_center, post_crop_height, post_crop_width), post_resize*).  Postprocessing takes place after augmentation.  
**permute:** supported values (*true, false*). If true then the data is shuffled every epoch.  
**input_img_height, input_img_width:** the final image height and width that are fed into the network. Resize operation uses these dimensions.  

##### VALIDATE_DATA section:
similar to TRAIN_DATA section. Usually set *augment=false, permute=false*.  

##### TEST_DATA section:
similar to VALIDATE_DATA section. Only used in *mode=test*.  
If testing with super patches that should be divided in 8x8 sub patches use *class_name = TCGASuperpatchBatchDataProvider*

##### TRAINER section:
**class_name:**  currently supported values are (*ClassifierTrainer*)  
set other training configuration values according to user preference, like batch size, learning rate, etc...  

##### TESTER section:
**class_name:**  
*ClassifierTesterExternalInput* :  if providing external input and not using a dataprovider (i.e. test data provider class name is set to 'None') and expected output is probability.  [Typically run using NNFramework_TF_external_call/external_model.py]

*ClassifierTesterExternalInputBinaryOutput* :  if providing external input and not using a dataprovider (i.e. test data provider class name is set to 'None') and expected output is thresholded probability (i.e. binary) and set the threshold.  [Typically run using NNFramework_TF_external_call/external_model.py]

*ClassifierTesterSuperpatchBatch*: Use with large patches that are further divided in sub-patches. Each row in the output will have the results of all the sub patches. Set batch_size =1 [Typically run using NNFramework_TF_external_call/external_train.py]

*ClassifierTesterBatch*: Use otherwise [Typically run using NNFramework_TF_external_call/external_train.py]


**out_dir:**  the location where test output files will be saved
**batch_size:**  desired batch size. Set according to preference and tester class used
If testing with super patches that should be divided in sub patches use batch_size =1
**threshold:**  Set according to desired threshold for binary prediction output, default threshold = 0.5

### Customization:
To create different classes such as for CNN architecture, dataprovider, trainer, or tester they should be supported by the runner file used to execute the training.  
Current supported runner file is: *tf_classifier_runner.py*   
for external input (i.e. not using a data provider) use *tf_classifier_runner_external_input.py* 

### Libraries used:
* Python 3.5 
* Tensorflow 1.8 with CUDA support
* Scikit-image
* The basic models used Resnet, Inception, and Inception-Resnet are imported from the TensorFlow-Slim image classification model library: https://github.com/tensorflow/models/tree/master/research/slim


### Running command:
To run use a command with the following format:  
python < runner py file > < config file full path >  

*Example:*  
python sa_runners/tf_classifier_runner.py $HOME/NNFramework/config/config_tcga_resnet-101.ini  



### Running batch jobs with Singularity  
#### To create a writable Singularity container:  
Build a sandbox to have a writable container to be able to install scikit-image and any future required libraries:  
module load singularity/3.0.0  
export SINGULARITY_DOCKER_USERNAME='$oauthtoken'  
export SINGULARITY_DOCKER_PASSWORD=< token >  
singularity build --sandbox tf-18.11-py3-w docker://nvcr.io/nvidia/tensorflow:18.11-py3  
  
#### Load and run inside the sandbox Singularity container:  
module unload singularity  
module load singularity/2.6.1  
singularity run --writable  --bind $SCRATCH --nv $SCRATCH/containers/tensorflow/tf-18.11-py3-w  
cd /home/shahira/NNFramework_TF_external_call 
python external_train.py $HOME/NNFramework/config_tcga_resnet-101.ini

external_model.py provides api for running from another application without using a data provider.
  

  


