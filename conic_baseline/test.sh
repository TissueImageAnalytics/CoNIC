
# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
LOCAL_INPUT=""
LOCAL_OUTPUT=""
# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! DO NOT MODIFY IF YOU ARE NOT CLEAR ABOUT DOCKER ENGINE
# <<<<<<<<<<<<<<<<<<<<<<<<<
./build.sh

# https://grand-challenge.org/documentation/data-storage/
# Docker Engine: this is different from the docker image to be run,
# this is the management system on the host machine that is going
# to run the docker

# Create a docker image named `conic-inference` then start running it.
# To run it, docker allocates 32GB memory (`--memory`) and allows
# using all GPU on the host (`gpus all`).
#
# Docker Engine then mounts 2 volumes (can either be virtual docker
# hard drive, physical hard drive, or local directory on the hosts)
# at these two location for IO:
# - /input/ : Containing any input data and configurations that the
# code within the docker image needs to run.
# - /output/ : Containing any output data that the docker will output.
# More Details: https://docs.docker.com/storage/volumes/
#
# After it finishes running, the docker image named `conic-inference` is removed.

docker run \
        --rm \
        --gpus all \
        --memory=32g \
        --mount src="$LOCAL_INPUT/",target=/input/,type=bind \
        --mount src="$LOCAL_OUTPUT/",target=/output/,type=bind \
        conic-inference
# >>>>>>>>>>>>>>>>>>>>>>>>>
