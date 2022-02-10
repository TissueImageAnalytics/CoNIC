#!/usr/bin/env bash

# TODO: Discuss with the team + Grand Challenge
# Depending on how Grand Challenge saving data within their mount volume,
# we can either import (copy over) script and runtime restriction
# or we have to download over from the internet (not ideal, as this implies
# user can modify the docker build and in total control of their runtime)
# may be AWS can set it up (???). At the very least, user should not be
# able to modify this script. Hence, we can still clone the conic repos
# for runtime restriction

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

printf "\nRun Path: $SCRIPTPATH\n"
printf "\n======================================\n"

./build.sh

docker volume create conic-output

# https://grand-challenge.org/documentation/data-storage/
# Docker Engine: this is different from the docker image to be run,
# this is the management system on the host machine that is going
# to run the docker

# Create a docker image named `conic` then start running it.
# To run it, docker allocates 32GB memory (`--memory`) and allows
# using all GPU on the host (`gpus all`).
#
# Docker Engine also creates two volumes within the user docker:
# - 1 created and managed by the Docker Engine.
# - 1 is already on the host system, but will be mounted to the
# target docker.
# More Details: https://docs.docker.com/storage/volumes/
#
# After finish running, the docker image named `conic` is removed.

# TODO: test run on grand challenge and change mount point
# docker run \
#         --rm \
#         --gpus all \
#         --memory=32g \
#         --mount src="$SCRIPTPATH/test/",target=/input/,type=bind \
#         --mount src=conic-output,target=/output/,type=volume \
#         conic
docker run \
        --rm \
        --gpus all \
        --memory=32g \
        --mount src="/mnt/storage_0/workspace/nuclei/conic-challenge/exp_output/local/data/valid",target=/input/,type=bind \
        --mount src=conic-output,target=/output/,type=volume \
        conic-inference
printf "\n======================================\n"
printf "Finish Running Inference\n"
printf "\n"

docker volume rm conic-output
