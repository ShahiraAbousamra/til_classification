## Docker description

This document explains how to build and run a docker image to generate whole slide image prediction heatmaps.

## CUDA versions

The base image for this container is nvcr.io/nvidia/tensorflow:18.11-py3. You will need CUDA version 10.x and driver version 410.xx on the host system (https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel\_18.11.html) to run the container. 

### Building the docker:

1. Navigate to the directory containing the Dockerfile: 
til\_classification/dockerfile

2. Run the docker build command:
docker build -t til\_pipeline\_frontiers .

### Running the docker:
#### 1. Required Folders Description:
**svs:** This folder will hold the .svs files to be processed.  
**patches:** The tiled whole slide image patches will be placed in this folder. After the patch extraction step it will contain subfolders with the filenames of the svs files extracted. The patches will be re-used when run on the same svs file multiple times. Otherwise can be deleted to save space.  
**output:** This folder will hold the log and final output folders. The pipeline will create the following subfolders to hold the output:  
  *heatmap\_txt:* will contain the text formatted output predictions.  
  *heatmap\_jsons:* will contain the json formatted output predictions.  
  *heatmap\_txt\_binary:* will contain the text formatted binary (thresholded) predictions.   
  *heatmap\_json\_binary:* will contain the json formatted binary (thresholded) predictions.  
  *log:* will contain the output log files.  

#### 2. Required Environment Variable Settings:
**MODEL\_CONFIG\_FILENAME:** The name of the model configuration file. There are 2 main configurations available:  
  *config\_incep-mix-new3\_test\_ext.ini:* The Inception-V4 trained model.  
  *config\_vgg-mix-new3\_test\_ext.ini:* The VGG-16 trained model.  
  *config\_resnet-mix-new3\_test\_ext.ini:* The ResNet-34 trained model.  

**CUDA\_VISIBLE\_DEVICES:** The gpu device to use. Default is '0'.  

**HEATMAP\_VERSION\_NAME:** The version name given to the set of predictions.  

**BINARY\_HEATMAP\_VERSION\_NAME:** The version name given to the set of binary (thresholded) predictions.  


#### 3. Optional Environment Variable Settings:
**LYM\_PREDICTION\_BATCH\_SIZE:** The batch size used in prediction. Default is 96. Can reduce it to overcome out of memory issues if any.  

#### 4. Command to Execute:
There are several scripts that are useful for generating predictions heatmaps from whole slide images. Replace the placeholder *{Command}* with any of the script filenames :

a. **cleanup\_heatmap.sh**  
Predictions are re-used if available. You can skip this step if you are only continuing the prediction without changing the model configuration.  
To create new predictions on the same svs files (probably using a different model) it is important to clean the previously generated predictions before performing running again using another model. This script also cleans the log folder.

b. **svs\_2\_heatmap.sh**  
Runs the heatmap generation *including* patch extraction (tiling)

c. **patch\_2\_heatmap.sh**  
Runs the heatmap generation *excluding* patch extraction (tiling)

d. **threshold\_probability\_heatmaps.sh**  
Creates binary predictions probability maps using the predefined thresholds saved in the model configuration file.  
This script does not run CNNs on GPUs.

e. **get\_binary\_png\_heatmaps.sh**  
Generate binary heatmaps in png format.  
This script does not run CNNs on GPUs.

  

#### 5. Execute:
Run the below command replacing the {placeholders} with appropriate settings:  

nvidia-docker run --name test\_til\_pipeline  -it \\  
-v *{svs folder path}*:/quip\_app/til\_classification/u24\_lymphocyte/data/svs  \\  
-v *{patches folder path}*:/quip\_app/til\_classification/u24\_lymphocyte/data/patches   \\  
-v *{output folder path}*:/quip\_app/til\_classification/u24\_lymphocyte/data/output   \\  
-e MODEL\_CONFIG\_FILENAME='*{model config file name}*'  \\  
-e CUDA\_VISIBLE\_DEVICES='*{GPU ID}*'  \\  
-e HEATMAP\_VERSION\_NAME='*{heatmap version name}*'  \\  
-e BINARY\_HEATMAP\_VERSION\_NAME='*{heatmap version name}*'  \\  
-e LYM\_PREDICTION\_BATCH_SIZE='*{batch size}*'  \\  
-d til\_pipeline\_frontiers:latest bash *{Command}*
 

**This is an example command with some settings:**  

nvidia-docker run --name test\_til\_pipeline --rm -it \\  
-v /nfs/bigbrain/shahira/svs:/quip\_app/til\_classification/u24\_lymphocyte/data/svs  \\  
-v /nfs/bigbrain/shahira/patches:/quip\_app/til\_classification/u24\_lymphocyte/data/patches   \\  
-v /nfs/bigbrain/shahira/til_output:/quip\_app/til\_classification/u24\_lymphocyte/data/output   \\  
-e MODEL\_CONFIG\_FILENAME='config\_vgg-mix\_test\_ext.ini'  \\  
-e CUDA\_VISIBLE\_DEVICES='0'  \\  
-e HEATMAP\_VERSION\_NAME='lym\_vgg-mix\_probability'  \\  
-e BINARY\_HEATMAP\_VERSION\_NAME='lym\_vgg-mix\_binary'  \\  
-e LYM\_PREDICTION\_BATCH\_SIZE=96  \\  
-d til\_pipeline\_frontiers:latest \\  
bash svs\_2\_heatmap.sh


Note that for *threshold\_probability\_heatmaps.sh* and *get\_binary\_png\_heatmaps.sh*, you can omit "-e CUDA\_VISIBLE\_DEVICES", "-e LYM\_PREDICTION\_BATCH\_SIZE" arguments, since these two scripts do not run CNN models on GPUs. Additionally, for *get\_binary\_png\_heatmaps.sh*, you can also omit "-e MODEL\_CONFIG\_FILENAME".

