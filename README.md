# Deep Learning-Based Mapping of Tumor Infiltrating Lymphocytes in Whole Slide Images of 23 Types of Cancer  
 
#### Shahira Abousamra, Rajarsi Gupta, Le Hou, Rebecca Batiste, Tianhao Zhao, Anand Shankar, Arvind Rao, Chao Chen, Dimitris Samaras, Tahsin Kurc, and Joel Saltz, Frontiers in Oncology, 2022.
 
This is the code associated with our journal. 
A deep learning workflow to classify 50x50 um tiled image patches (100x100 pixels at 20x magnification) as TIL positive or negative based on the presence of TILs in gigapixel whole slide images (WSIs) from the Cancer Genome Atlas (TCGA). This workflow generates TIL maps to study the abundance and spatial distribution of TILs in 23 different types of cancer.

**WSI Prediction:** 
The code for TIL map prediction on WSIs is in the folder: `u24\_lymphocyte`.
To run it through docker, use the docker file: `Dockerfile`.
Instructions for running the docker are availabel in the folder: `docker\_readme`.

**Source codes for the tensor flow models:** 
are in the folder `src_incep_vgg_tf`.

**Source codes for processing predictions on test data:** 
are in the folder `process_results`.


 