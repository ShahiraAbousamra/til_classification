#!/bin/bash

if [[ -n $BASE_DIR ]]; then
	cd $BASE_DIR
else
	cd ../
fi

source ./conf/variables.sh

in_dir=${BASE_DIR}

rm ${HEATMAP_TXT_OUTPUT_FOLDER}/*
rm ${JSON_OUTPUT_FOLDER}/*
rm ${BINARY_HEATMAP_TXT_OUTPUT_FOLDER}/*
rm ${BINARY_JSON_OUTPUT_FOLDER}/*
rm ${LOG_OUTPUT_FOLDER}/*

delete_pattern=patch-level-lym.txt
for folder in ${PATCH_PATH}/*; do
	echo $folder
	find $folder -name $delete_pattern -delete
	#rm "$folder/$delete_pattern"
done

wait;

exit 0

