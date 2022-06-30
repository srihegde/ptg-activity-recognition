#! /bin/sh

# Set default names for image and container
imageName=mmdetection3d
containerName=ho_features

# Build docker
docker build -t $imageName -f docker/Dockerfile .
