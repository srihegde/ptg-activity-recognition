#! /bin/sh

# Run docker
DATA_VOLUME_SRC="/home/local/KHQ/sri.hegde/kitware/activity_recognition/datasets/h2o"
DATA_VOLUME_DST="/mmdetection3d/data/h2o"
docker run -d\
	 -p 5000:5000\
	--gpus all\
	 --shm-size=8g\
	  -it -v $DATA_VOLUME_SRC:$DATA_VOLUME_DST mmdetection3d