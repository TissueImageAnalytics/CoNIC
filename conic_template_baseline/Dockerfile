# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# Edit the base image here, e.g., to use 
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/) 
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
# For running opencv-python
RUN apt-get update
RUN apt-get install gcc ffmpeg libsm6 libxext6 openslide-tools -y

# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! DO NOT MODIFY
# <<<<<<<<<<<<<<<<<<<<<<<<<
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! DO NOT MODIFY - ORGANIZER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# Copy all required files such that they are available within the docker image (code, weights, ...)

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# Copy all required files such that they are available within the docker image (code, weights, ...)
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm data/ /opt/algorithm/data/
COPY --chown=algorithm:algorithm source/ /opt/algorithm/source/

# Install required python packages via pip - you may adapt the requirements.txt to your needs
RUN python -m pip install --user -r requirements.txt

# Entrypoint to your python code - executes process.py as a script
ENTRYPOINT python -m process $0 $@
# >>>>>>>>>>>>>>>>>>>>>>>>>

## ALGORITHM LABELS ##

# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=ConicTemplate
# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! DO NOT MODIFY - ORGANIZER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=16G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=11.1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=12G
# >>>>>>>>>>>>>>>>>>>>>>>>>
