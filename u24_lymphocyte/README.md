# u24_lymphocyte

This software implements the pipeline for the lymphocyte classification project. 

List of folders and functionalities are below: 

scripts/: contains scripts that connect several sub-functionalities together for complete functionalities such as patch extraction and prediction.

conf/: contains configuration. 

data/: a place where should contain all logs, input/output images, and large files. 

patch_extraction/: extracts all patches from svs images. Mainly used in the test phase. 

prediction/: CNN prediction code. 

heatmap_gen/: generate json files that represents heatmaps for quip, using the lymphocyte and necrosis CNNs' raw output txt files. 

