#!/bin/bash -f 

source ../../conf/variables.sh

# replace /root in the model ini files
for i in `ls *.ini`; do 
	sed -i 's/\/root/\'"${APP_DIR}"'/g' $i;
done

exit 0;