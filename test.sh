#!/usr/bin/env bash
export CONTAINERD_ENABLE_DEPRECATED_PULL_SCHEMA_1_IMAGE=1

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# ./build.sh

# VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut -c 1-10)

# DOCKER_FILE_SHARE=picai_baseline_nnunet_processor-output-$VOLUME_SUFFIX
# docker volume create $DOCKER_FILE_SHARE
DOCKER_FILE_SHARE=$(pwd)/output_debug
mkdir -p $DOCKER_FILE_SHARE
# you can see your output (to debug what's going on) by specifying a path instead:
# DOCKER_FILE_SHARE="/mnt/netcache/pelvis/projects/joeran/tmp-docker-volume"

docker run --cpus=4 --memory=32gb --shm-size=32gb --rm \
        -v $SCRIPTPATH/test/:/input/ \
        -v $DOCKER_FILE_SHARE:/output/ \
        picai_baseline_nnunet_processor

# check detection map (at /output/images/cspca-detection-map/cspca_detection_map.mha)
# check detection map (using Python 3.10 instead of deprecated SimpleITK Docker image)
docker run --rm \
    -v $DOCKER_FILE_SHARE:/output/ \
    -v $SCRIPTPATH/test/:/input/ \
    python:3.10-slim bash -c "pip install -q SimpleITK numpy && python -c \"import sys, numpy as np, SimpleITK as sitk; f1 = sitk.GetArrayFromImage(sitk.ReadImage('/output/images/cspca-detection-map/cspca_detection_map.mha')); f2 = sitk.GetArrayFromImage(sitk.ReadImage('/input/cspca-detection-map/10032_1000032.mha')); diff = np.sum(np.abs(f1-f2)>1e-3); print('N/o voxels more than 1e-3 different between prediction and reference:', diff); sys.exit(int(diff > 10))\""

if [ $? -eq 0 ]; then
    echo "Detection map test successfully passed..."
else
    echo "Expected detection map was not found..."
fi

# check case_confidence (at /output/cspca-case-level-likelihood.json)
# check case_confidence (using Python 3.10 instead of deprecated SimpleITK Docker image)
docker run --rm \
    -v $DOCKER_FILE_SHARE:/output/ \
    -v $SCRIPTPATH/test/:/input/ \
    python:3.10-slim bash -c "pip install -q numpy && python -c \"import sys, json; f1 = json.load(open('/output/cspca-case-level-likelihood.json')); f2 = json.load(open('/input/cspca-detection-map/10032_1000032.json')); print(f'Found case-level prediction {f1}, expected {f2}'); sys.exit(int(abs(f1-f2) > 1e-3))\""


if [ $? -eq 0 ]; then
    echo "Case-level prediction test successfully passed..."
else
    echo "Expected case-level prediction was not found..."
fi

# docker volume rm $DOCKER_FILE_SHARE

