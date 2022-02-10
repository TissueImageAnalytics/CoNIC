<p align="center">
  <img src="/doc/conic_banner.png">
</p>

# Blank template for docker submission of CoNIC challenge

This repository contains everything participants need to dockerize their algorithms for CoNIC challenge. In this README, you will learn about this repository structure and how you should use it for your submissions to CoNIC challenge. There are two main sections covered in this guide:

- [Algorithm dockerization](#dockerize-your-algorithm)
- [Dockerized algorithm submission](#submit-your-algorithm)

As we only accept the dockerized algorithm submissions on CoNIC challenge and because the Grand-Challenge paltform (challenge host) only works with dockerized containers that have a cetrain structure, you need to follow the guidelines in this document to have a successful submission to the challenge. General guidelines are the same for both "Task 1: Nuclear segmentation and classification" and "Task 2: Prediction of cellular composition" parts of the CoNIC challenge and can be used for both "Preliminary" and "Final" test phases (read more about challenge formate [here](https://conic-challenge.grand-challenge.org/)).


## Dockerize Your Algorithm

The following steps should be done to create an acceptable dockerized algorithm for the challenge.

### 1- Prerequisite
To create and test your docker setup, you will need to install [Docker](https://docs.docker.com/engine/install/)
and [NVIDIA-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (in case you need GPU computation).

After installing the docker, you can start by cloning this repository (bare in mind that this is directory is in `docker-template` branch of the `CoNIC` repository):
```
git clone -b docker-template https://github.com/TissueImageAnalytics/CoNIC
```
There are two different docker templates in that repository: 
- `conic_template_blank`: This directory contains a docker template with all the essential functions and modules needed to create an acceptable docker container for the CoNIC challenge. 
- `conic_template_baseline`: This directory contains all necessary files available in the `conic_template_blank` as well as some other python functions and modules related to the [CoNIC baseline method](https://github.com/vqdang/hover_net/tree/conic).

In this example, we are focusing on the `conic_template_blank`.


### 2- Embedding algorithm into the docker template

#### Main files in the docker template
In this repository we provide a template to construct your own docker for submission.
You will need to edit the following files:

- `source/main.py`: This file contains a function `run.py` with its API defined
    so that **Organizers** and Grand Challenge system can pick up your predictions
    for evaluation.

- `process.py`: You should only edit `LOCAL_ENTRY` and `EXECUTE_IN_DOCKER`
    to run within your debugger (terminal, pycharm or vscode) or within the docker.

- `requirements.txt`: This file contains all python libraries that are required to
    run your code

Typically, we expect you to copy your entire project folder over and put it under
`source` directory. As for any auxiliary data (model weights, stain normalization
target, etc.), you should put it under `data` directory.

#### Expected outputs
 - Task 1:
 - Task 2:

### 3- Creating the docker container

### 4- Testing the docker container
Once you have verified they are installed correctly.
You can test your docker by doing:

```bash
sudo ./test.sh
```

### 5- Exporting the docker container

===========================================================================

## Submit Your Algorithm

Assuming you have registered for the challenge and read the challenge rules.

### 1- Uplaod your algorithm

### 2- Submit your algorithm
