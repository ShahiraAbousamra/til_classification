#!/bin/bash

if [[ -n $BASE_DIR ]]; then
    cd $BASE_DIR
else
    cd ../
fi
source ./conf/variables.sh
cd ./scripts

# Path contains the svs slides
# This is just used for getting the height and width
# of the slides
SLIDES=${SVS_INPUT_PATH}

# Locations of unmodified heatmaps
# The filenames of the unmodifed heatmaps should be:
#   prediction-${slide_id}
# For example:
#   prediction-TCGA-NJ-A55O-01Z-00-DX1
mkdir -p png_folder
rm -rf ./png_folder/*
PNG_FOLDER=`readlink -e png_folder`

for files in ${HEATMAP_TXT_OUTPUT_FOLDER}/color-*; do
    # Get slide id
    SVS=`echo ${files} | awk -F'/' '{print $NF}' | awk -F'color-' '{print $2}'`

    # Find the unmodified heatmap
    PRED=`ls -1 ${BINARY_HEATMAP_TXT_OUTPUT_FOLDER}/prediction-${SVS}*|grep -v low_res`
    COLOR=${files}

    SVS_FILE=`ls -1 ${SLIDES}/${SVS}*.??? | head -n 1`
    if [ -z "$SVS_FILE" ]; then
        SVS_FILE=`ls -1 ${SLIDES}/${SVS}*.???? | head -n 1`
        if [ -z "$SVS_FILE" ]; then
            echo "Could not find slide."
            continue;
        fi
    fi

    WIDTH=` openslide-show-properties ${SVS_FILE} \
          | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SVS_FILE} \
          | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    python ../download_heatmap/get_thresholded_heatmaps/get_thresholded_heatmap.py \
        ${SVS} ${WIDTH} ${HEIGHT} ${PRED} ${COLOR} ${PNG_FOLDER}
done

mkdir -p ${THRESHOLDED_HEATMAPS_PATH}
rm -rf ${THRESHOLDED_HEATMAPS_PATH}/*.png
mv ./png_folder/* ${THRESHOLDED_HEATMAPS_PATH}/

exit 0
