<p align="center">
  <img src="/doc/conic_banner.png">
</p>

# Blank template for docker submission of CoNIC challenge

This repository contains everything participants need to dockerize their algorithms for CoNIC challenge. In this README, you will learn about this repository structure and how you should use it for your submissions to CoNIC challenge. There are two main sections covered in this guide:

- [Dockerize Your Algorithm](#dockerize-your-algorithm)
- [Submit Your Algorithm](#submit-your-algorithm)

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

#### Main files and folders in the docker template
Here are important files and subfolder that you should include in your algorithm docker and these should be always there:

- `process.py`:  This is the main file that will be run by the Grand-Challenge platform when you submit your algorithm. We made it easy for you and don't need to change anything in this file except for `LOCAL_ENTRY` and `EXECUTE_IN_DOCKER` parameters if you wish to test your algorithm on your local machine. In that case, you need to set `EXECUTE_IN_DOCKER = True` and change the the `LOCAL_ENTRY` dictionary values (for `"input_dir"`, `"output_dir"`, and `user_data_dir` keys) according to your system. When `process.py` is run, it calls `run` function from your source algorithm (explained below).

- `source/main.py`: This file contains a function `run` with its API defined
    so that organizers and Grand Challenge system can pick up your predictions
    for evaluation. **Important**: we expect you to copy your entire project folder to the `source` directory and refactor it in a way that there will be a `main.py` module with `run` function in it which is responsible for processing all the images in the test set and save the results. In the template, we have set the way that input data needs to be read from the Grand-Challenge platform as numpy array in form of [Nx256x256x3]. You are responsible to fill the `run` function with your algorithm calls to process the input images and create the expected outputs for each task (we explain the expected outputs for each task in [Expected outputs](#expected-outputs) section).

- `data` directory: Every external data users needs to run their algorithms should be put in this directory. Such auxiliary data can be model trained weights, stain normalization
target, etc. **Important**: Please note that your docker container will not have access to interent when it's being run on Grand-Challenge platform. Therefore, all data and packages required for the algorithm should be included in the docker container upon its creation.


There are also some files related to docker creation, testing, and exporting that will be explained later.

#### Expected outputs
It is important to fully understand the expected format of algorithm output for each task. As you know, the input to algorithms for both tasks is the same and is a numpy array of shape `(N, 256, 256, 3)` where `N` is the number of test images. The code snippet needed to load this data from Grand-Challenge platform as a numpy image is given in the template code. Also, we tried to include the code needed to convert algorithm outputs and saving results to disk for each task.

 - Task 1 - Nuclear segmentation and classification: In this task participant are expected to process all the images in the input array and create another array with the shape of  `(N, 256, 256, 2)` where `N` is the number of test images and 2 channels are for predicted instance map and class map. This array needs to be converted and saved in the `/output/` directory as an `.mha` file. A code snippet to do so is provided in the template of `main.py`.

 - Task 2 - Prediction of cellular composition: In this task participant are expected to generate JSON file for each type of cell which contains a list of the number of cells of that type in each image. Assuming the algorithm calculates a list of `N` items where item `n` in that list contains cellular composition of image at index `n` in the input (`input_array[n]`), there is template code in `main.py` which extract the cellular composition of each cell type and save it with appropriate JSON format.

### 3- Creating the docker container
Once you have your algorithm ready according to the challenge template, you can use the `Dockerfile` template in the repository to containerize your algorithm. All the instruction you need are commented in the file, however, we explain the important parts of the files you need to change when creating your docker container:

- `Dockerfile`: This is the main file responsible for docker configuration. Please make sure not changing the code parts specified by `"# ! DO NOT MODIFY"` as these are preset configurations agreed to be used with Grand-challenge platform. However, you might need to `"# ! USER SPECIFIC"` regions. For example, you need to include your desired base image and linux packages to be installed at the top of the `Dockerfile`. It is recommended to install your desired python packages using the list `requirements.txt` list (explained below) instead of installing them one-by-one whithin the `Dockerfile`.
- `requirements.txt`: This file contains a list of all python libraries that are required to run your code. Please make sure that you include every thing you need and this can be checked by testing your docker build locally (explained in the next section).
- `build.sh`: Helper bash script to generate a docker container based on the provided `Dockerfile` in the directory. Remember, in order to run this script you need to have a working installation of Docker on you system.

Basically, there are three easy steps to create your docker container when you have algorithm ready: First, you need to modify the `Dockerfile` and `requirements.txt` accourding to your needs and then run the `build.sh` bash script in your terminal to create the container:
```bash
sudo ./build.sh
```

### 4- Testing the docker container
Once you have created your docker container, you can verify if it's working as expected by running the provided `test.sh` bash script:
```bash
sudo ./test.sh
```
This bash script first tries to build the docker container (if it hasn't been built yet) by calling `build.sh` internally and then running that container by providing the information it needs. Particularly, we require to mount two volumes one the docker: one created and managed by the docker engine (to save the results in) and another one should be already on the host system (which container input images to the docker). If you want to test your container locally, you have to set the path to the directory which will be mounted to `target=/input/` in the container:
```bash
docker run \
        --rm \
        --gpus all \
        --memory=32g \
        --mount src="path_to_the_directory_with_test_image.mha",target=/input/,type=bind \
        --mount src=conic-output,target=/output/,type=volume \
        conic-inference
```

The docker container, similar to Grand-Challenge platform, expects the input to be in form a `.mha` image file. To be able to test your docker containers, we have provided a sample `.mha` file [here]() so you can download and and put in your desired directory to be used for testing. If the testing goes well, dockerized algorithm should save the output in the created `conic-output` volume or any other user-specific results should be shown on the screen. Note that `--memory` argument should be set based on your system specifications.

### 5- Exporting the docker container


===========================================================================

## Submit Your Algorithm

Assuming you have registered for the challenge and read the challenge rules.

### 1- Uplaod your algorithm

### 2- Submit your algorithm
