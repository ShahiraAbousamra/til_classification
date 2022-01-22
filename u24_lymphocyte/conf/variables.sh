#!/bin/bash

# Variables
DEFAULT_OBJ=40
DEFAULT_MPP=0.25
MONGODB_HOST=xyz
MONGODB_PORT=27017
CANCER_TYPE=all

if [[ -n $HEATMAP_VERSION_NAME ]]; then
	export HEATMAP_VERSION=$HEATMAP_VERSION_NAME ;
else
	export HEATMAP_VERSION=lym_incep4_mix_new_prob ;
fi
if [[ -n $BINARY_HEATMAP_VERSION_NAME ]]; then
	export BINARY_HEATMAP_VERSION=$BINARY_HEATMAP_VERSION_NAME ;
else
	export BINARY_HEATMAP_VERSION=lym_incep4_mix_new_binary ;
fi
if [[ ! -n $LYM_PREDICTION_BATCH_SIZE ]]; then
   export LYM_PREDICTION_BATCH_SIZE=96;
fi

# Base data and output directories
export APP_DIR=/quip_app
export BASE_DIR=${APP_DIR}/til_classification
export TIL_DIR=${BASE_DIR}/u24_lymphocyte/
export DATA_DIR=${TIL_DIR}/data
export OUT_DIR=${DATA_DIR}/output

# Prediction folders
# Paths of data, log, input, and output
export SVS_INPUT_PATH=${DATA_DIR}/svs
export JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons
export BINARY_JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons_binary
export HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt
export BINARY_HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt_binary
export LOG_OUTPUT_FOLDER=${OUT_DIR}/log
export PATCH_PATH=${DATA_DIR}/patches
export OUT_FOLDERS="${JSON_OUTPUT_FOLDER} ${BINARY_JSON_OUTPUT_FOLDER} ${HEATMAP_TXT_OUTPUT_FOLDER} ${BINARY_HEATMAP_TXT_OUTPUT_FOLDER} ${LOG_OUTPUT_FOLDER} ${PATCH_PATH}"

# Trained model
if [[ -n $MODEL_CONFIG_FILENAME ]]; then
  export LYM_NECRO_CNN_MODEL_PATH=${TIL_DIR}/prediction/models/${MODEL_CONFIG_FILENAME} ;
else
  export LYM_NECRO_CNN_MODEL_PATH=${TIL_DIR}/prediction/models/config_incep-mix-new3_test_ext.ini ;
fi
EXTERNAL_LYM_MODEL=1

# VERSION INFO
export MODEL_PATH=$LYM_NECRO_CNN_MODEL_PATH
export TIL_VERSION=$(git show --oneline -s | cut -f 1 -d ' ')":"$MODEL_VER":"$(sha256sum $MODEL_PATH | cut -c1-7)
export GIT_REMOTE=$(git remote -v | head -n 1 | cut -f 1 -d ' '| cut -f 2)
export GIT_BRANCH=$(git branch | grep "\*" | cut -f 2 -d ' ')
export GIT_COMMIT=$(git show | head -n 1 | cut -f 2 -d ' ')
export MODEL_HASH=$(sha256sum $LYM_NECRO_CNN_MODEL_PATH | cut -f 1 -d ' ')


# create missing output directories
if [ ! -d ${OUT_DIR} ]; then
  mkdir ${OUT_DIR} ;
fi

if [ ! -d ${JSON_OUTPUT_FOLDER} ]; then
  mkdir ${JSON_OUTPUT_FOLDER} ;
fi

if [ ! -d ${BINARY_JSON_OUTPUT_FOLDER} ]; then
  mkdir ${BINARY_JSON_OUTPUT_FOLDER} ;
fi

if [ ! -d ${HEATMAP_TXT_OUTPUT_FOLDER} ]; then
  mkdir ${HEATMAP_TXT_OUTPUT_FOLDER} ;
fi

if [ ! -d ${BINARY_HEATMAP_TXT_OUTPUT_FOLDER} ]; then
  mkdir ${BINARY_HEATMAP_TXT_OUTPUT_FOLDER} ;
fi

if [ ! -d ${LOG_OUTPUT_FOLDER} ]; then
  mkdir ${LOG_OUTPUT_FOLDER} ;
fi

if [ ! -d ${PATCH_PATH} ]; then
  mkdir ${PATCH_PATH} ;
fi



# Base directory
#BASE_DIR=/data03/shared/sabousamra/u24_lymphocyte_clean
#OUT_DIR=/data03/shared/sabousamra/quip_classification_output

# Paths of data, log, input, and output
#JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons
#BINARY_JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons_binary
#HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt
#BINARY_HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt_binary
#LOG_OUTPUT_FOLDER=${OUT_DIR}/log
#SVS_INPUT_PATH=/data03/shared/sabousamra/TCGA_PAAD_DX1
#PATCH_PATH=/data03/shared/sabousamra/TCGA_PAAD_DX1_patches


# LYM_NECRO_CNN_MODEL_PATH=/gpfs/projects/KurcGroup/sabousamra/TIL_classification/models/NNFramework_TF_models/config_incep-mix-new3_test_ext.ini
#LYM_NECRO_CNN_MODEL_PATH=/gpfs/projects/KurcGroup/sabousamra/TIL_classification/models/config_resnet-mix-new3_test_ext.ini
#LYM_NECRO_CNN_MODEL_PATH=/gpfs/projects/KurcGroup/sabousamra/TIL_classification/models/NNFramework_TF_models/config_vgg-mix-new3_test_ext.ini
# if [[ -n $MODEL_CONFIG_FILENAME ]]; then
  # LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/${MODEL_CONFIG_FILENAME} ;
# else
  # LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/config_vgg-mix_test_ext.ini ;
# fi
# if [[ -n $LYM_PREDICTION_BATCH_SIZE ]]; then
  # LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/${MODEL_CONFIG_FILENAME} ;
# else
  # LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/config_vgg-mix_test_ext.ini ;
# fi

# if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
	# export LYM_CNN_PRED_DEVICE=0
# else
	# export LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICES}
# fi

