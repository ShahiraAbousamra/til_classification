## Docker description

This document explains how to build and run a docker image to generate whole slide image prediction heatmaps.

## CUDA versions

The base image for this container is nvcr.io/nvidia/tensorflow:18.11-py3. You will need CUDA version 10.x and driver version 410.xx on the host system (https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_18.11.html) to run the container. 

### Building the docker:

1. Navigate to the directory containing the Dockerfile: 
quip_classification/dockerfile

2. Run the docker build command:
docker build -t til_pipeline .

### Running the docker:
#### 1. Required Folders Description:
**svs:** This folder will hold the .svs files to be processed.  
**patches:** The tiled whole slide image patches will be placed in this folder. After the patch extraction step it will contain subfolders with the filenames of the svs files extracted. The patches will be re-used when run on the same svs file multiple times. Otherwise can be deleted to save space.  
**output:** This folder will hold the log and final output folders. The pipeline will create the following subfolders to hold the output:  
  *heatmap_txt:* will contain the text formatted output predictions.  
  *heatmap_jsons:* will contain the json formatted output predictions.  
  *heatmap_txt_binary:* will contain the text formatted binary (thresholded) predictions.   
  *heatmap_json_binary:* will contain the json formatted binary (thresholded) predictions.  
  *log:* will contain the output log files.  

#### 2. Required Environment Variable Settings:
**MODEL_CONFIG_FILENAME:** The name of the model configuration file. There are 2 main configurations available:  
  *config_incep-mix_test_ext.ini:* The inception-v4 model trained on mix of manual and semi-autoamted labels each from a different set of cancer types.  
  *config_vgg-mix_test_ext.ini:* The vgg-16 model trained on mix of manual and semi-autoamted labels each from a different set of cancer types.  

**CUDA_VISIBLE_DEVICES:** The gpu device to use. Default is '0'.  

**HEATMAP_VERSION_NAME:** The version name given to the set of predictions.  

**BINARY_HEATMAP_VERSION_NAME:** The version name given to the set of binary (thresholded) predictions.  


#### 3. Optional Environment Variable Settings:
**LYM_PREDICTION_BATCH_SIZE:** The batch size used in prediction. Default is 96. Can reduce it to overcome out of memory issues if any.  

#### 4. Command to Execute:
There are several scripts that are useful for generating predictions heatmaps from whole slide images. Replace the placeholder *{Command}* with any of the script filenames :

a. **cleanup_heatmap.sh**  
Predictions are re-used if available. You can skip this step if you are only continuing the prediction without changing the model configuration.  
To create new predictions on the same svs files (probably using a different model) it is important to clean the previously generated predictions before performing running again using another model. This script also cleans the log folder.

b. **svs_2_heatmap.sh**  
Runs the heatmap generation *including* patch extraction (tiling)

c. **patch_2_heatmap.sh**  
Runs the heatmap generation *excluding* patch extraction (tiling)

d. **threshold_probability_heatmaps.sh**  
Creates binary predictions probability maps using the predefined thresholds saved in the model configuration file.  
This script does not run CNNs on GPUs.

e. **get_binary_png_heatmaps.sh**  
Generate binary heatmaps in png format.  
This script does not run CNNs on GPUs.

  

#### 5. Execute:
Run the below command replacing the {placeholders} with appropriate settings:  

nvidia-docker run --name test_til_pipeline  -it \\  
-v *{svs folder path}*:/root/quip_classification/u24_lymphocyte/data/svs  \\  
-v *{patches folder path}*:/root/quip_classification/u24_lymphocyte/data/patches   \\  
-v *{output folder path}*:/root/quip_classification/u24_lymphocyte/data/output   \\  
-e MODEL_CONFIG_FILENAME='*{model config file name}*'  \\  
-e CUDA_VISIBLE_DEVICES='*{GPU ID}*'  \\  
-e HEATMAP_VERSION_NAME='*{heatmap version name}*'  \\  
-e BINARY_HEATMAP_VERSION_NAME='*{heatmap version name}*'  \\  
-e LYM_PREDICTION_BATCH_SIZE='*{batch size}*'  \\  
-d til_pipeline:latest  *{Command}*
 

**This is an example command with some settings:**  

nvidia-docker run --name til_pipeline --rm -it \\  
-v /nfs/bigbrain/shahira/svs:/root/quip_classification/u24_lymphocyte/data/svs  \\  
-v /nfs/bigbrain/shahira/patches:/root/quip_classification/u24_lymphocyte/data/patches   \\  
-v /nfs/bigbrain/shahira/til_output:/root/quip_classification/u24_lymphocyte/data/output   \\  
-e MODEL_CONFIG_FILENAME='config_vgg-mix_test_ext.ini'  \\  
-e CUDA_VISIBLE_DEVICES='0'  \\  
-e HEATMAP_VERSION_NAME='lym_vgg-mix_probability'  \\  
-e BINARY_HEATMAP_VERSION_NAME='lym_vgg-mix_binary'  \\  
-e LYM_PREDICTION_BATCH_SIZE=96  \\  
-d til_pipeline:latest \\  
svs_2_heatmap.sh


Note that for *threshold_probability_heatmaps.sh* and *get_binary_png_heatmaps.sh*, you can omit "-e CUDA_VISIBLE_DEVICES", "-e LYM_PREDICTION_BATCH_SIZE" arguments, since these two scripts do not run CNN models on GPUs. Additionally, for *get_binary_png_heatmaps.sh*, you can also omit "-e MODEL_CONFIG_FILENAME".

