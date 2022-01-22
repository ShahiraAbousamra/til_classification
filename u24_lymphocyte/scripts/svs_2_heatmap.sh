#!/bin/bash

cd ../

source ./conf/variables.sh

# create missing output directories
for i in ${OUT_FOLDERS}; do
	if [ ! -d $i ]; then
		mkdir -p $i;
	fi
done

cd patch_extraction
nohup bash start.sh &
cd ..

cd prediction
nohup bash start.sh &
cd ..

wait;

cd prediction
nohup bash start.sh &
cd ..

wait;

cd heatmap_gen
nohup bash start.sh &
cd ..

wait;

exit 0
