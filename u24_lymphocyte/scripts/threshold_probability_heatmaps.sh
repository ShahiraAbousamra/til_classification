#!/bin/bash

# reads the threshold value from the model config file and applies the threshold on heatmap json files located in $JSON_OUTPUT_FOLDER and heatmap txt files located in $HEATMAP_TXT_OUTPUT_FOLDER
# the thresholded versions are stored in $BINARY_JSON_OUTPUT_FOLDER and $BINARY_HEATMAP_TXT_OUTPUT_FOLDER respectively with the heatmap version name given by $BINARY_HEATMAP_VERSION

if [[ -n $BASE_DIR ]]; then
	cd $BASE_DIR
else
	cd ../
fi

source ./conf/variables.sh

cd ./scripts

python threshold_probability_heatmaps.py $JSON_OUTPUT_FOLDER  $BINARY_JSON_OUTPUT_FOLDER  $HEATMAP_TXT_OUTPUT_FOLDER  $BINARY_HEATMAP_TXT_OUTPUT_FOLDER   $LYM_NECRO_CNN_MODEL_PATH  $BINARY_HEATMAP_VERSION


wait;

