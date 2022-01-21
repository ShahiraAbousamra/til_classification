#!/bin/bash

if [[ -n $BASE_DIR ]]; then
	cd $BASE_DIR
else
	cd ../
fi
source ./conf/variables.sh


cd prediction
nohup bash start.sh &
cd ..

wait;

cd heatmap_gen
nohup bash start.sh &
cd ..

wait;

exit 0
