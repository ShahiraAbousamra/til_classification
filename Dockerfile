# FROM	nvcr.io/nvidia/tensorflow:18.11-py3
FROM tensorflow/tensorflow:1.13.1-gpu-py3
MAINTAINER til_classification 

RUN 	apt-get -y update && \
	apt-get -y install python3-pip openslide-tools wget git && \
	pip install openslide-python scikit-image pymongo && \ 
	pip install torch==1.0.1 torchvision==0.2.2 

ENV	BASE_DIR="/quip_app/til_classification"
ENV	PATH="./":$PATH

COPY	. ${BASE_DIR}/.

ENV     MODEL_VER="v2.0"
ENV	MODEL_URL="https://stonybrookmedicine.box.com/shared/static/r0p2h2oh53zw52o6fm6qrqjtcwd3p0ga.zip"

RUN	cd ${BASE_DIR}/u24_lymphocyte/prediction/models && \
	wget -v -O models.zip -L $MODEL_URL >/dev/null 2>&1 && \
    	unzip -o models.zip && rm -f models.zip && \
	bash ./update_path.sh && \
	chmod 0755 ${BASE_DIR}/u24_lymphocyte/scripts/*

WORKDIR ${BASE_DIR}/u24_lymphocyte/scripts

CMD ["/bin/bash"]