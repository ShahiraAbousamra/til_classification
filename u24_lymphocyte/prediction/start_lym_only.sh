#!/bin/bash

source ../conf/variables.sh

cd lymphocyte
nohup bash pred_thread_lym.sh \
    ${PATCH_PATH} 0 1 ${LYM_CNN_PRED_DEVICE} \
    &> ${LOG_OUTPUT_FOLDER}/log.pred_thread_lym_0.txt &
cd ..


wait

exit 0
