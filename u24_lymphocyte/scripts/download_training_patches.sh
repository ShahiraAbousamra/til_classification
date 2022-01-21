#!/bin/bash

cd ../

# 1. Query database and download markups in json format.
#
# Input:
#   download_heatmap/download_markings/list.txt
#     The list of WSIs Raj labeled
#   download_heatmap/download_markings/get_raw_json_mongoexport.sh
#     Change the annotator's accout name if he/she is not Raj.
# Output:
#   download_heatmap/download_markings/raw_json/
cd download_heatmap/download_markings/
if [ ! -f lists.txt ]; then
    echo "Error: no download_heatmap/download_markings/lists.txt"
    exit 1
fi
bash get_raw_json_mongoexport.sh
cd ../../

# 2. Convert json to easy txt format
cd download_heatmap/download_markings/
bash draw_raw_xy.sh
cd ../../

# 3. Get modified heatmap png files
cd download_heatmap/get_modified_heatmaps/
bash start.sh
cd ../../

# 4. Extract all labeled patches
cd additional_code/get_ground_truth/
bash get_patch_coordinates.sh
bash draw_patches.sh
cd ../../

# Results are under additional_code/get_ground_truth/patches/
exit 0
