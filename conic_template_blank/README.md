<p align="center">
  <img src="/doc/conic_banner.png">
</p>

# Blank template for docker submission of CoNIC challenge

This repository contains everything participants need to dockerize their algorithms and submission for the CoNIC challenge. In this README, you will learn about this repository structure and how you should use it for your submissions to the CoNIC challenge. There are two main sections covered in this guide:

- [Dockerize Your Algorithm](#dockerize-your-algorithm)
- [Submit Your Algorithm](#submit-your-algorithm)

As we only accept the dockerized algorithm submissions on the CoNIC challenge and because the Grand-Challenge platform (challenge host) only works with dockerized containers that have a certain structure, you need to follow the guidelines in this document to have a successful submission to the challenge. General guidelines are the same for both "Task 1: Nuclear segmentation and classification" and "Task 2: Prediction of cellular composition" parts of the CoNIC challenge and can be used for both "Preliminary" and "Final" test phases (read more about challenge format [here](https://conic-challenge.grand-challenge.org/)).


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
Here are important files and subfolders that you should include in your algorithm docker and these should be always there:

- `process.py`:  This is the main file that will be run by the Grand-Challenge platform when you submit your algorithm. We made it easy for you and don't need to change anything in this file except for `LOCAL_ENTRY` and `EXECUTE_IN_DOCKER` parameters if you wish to test your algorithm on your local machine. In that case, you need to set `EXECUTE_IN_DOCKER = True` and change the the `LOCAL_ENTRY` dictionary values (for `"input_dir"`, `"output_dir"`, and `user_data_dir` keys) according to your system. When `process.py` is run, it calls `run` function from your source algorithm (explained below).

- `source/main.py`: This file contains a function `run` with its API defined
    so that organizers and the Grand-Challenge system can pick up your predictions
    for evaluation. **Important**: we expect you to copy your entire project folder to the `source` directory and refactor it in a way that there will be a `main.py` module with `run` function in it which is responsible for processing all the images in the test set and save the results. In the template, we have set the way that input data needs to be read from the Grand-Challenge platform as NumPy array in form of [Nx256x256x3]. You are responsible to fill the `run` function with your algorithm calls to process the input images and create the expected outputs for each task (we explain the expected outputs for each task in the [Expected outputs](#expected-outputs) section).

- `data` directory: Every external data that users require to run their algorithms should be put in this directory. Such auxiliary data can be model trained weights, stain normalization
target, etc. **Important**: Please note that your docker container will not have interent access when it's being run on the Grand-Challenge platform. Therefore, all data and packages required for the algorithm should be included in the docker container upon its creation.


There are also some files related to docker creation, testing, and exporting that will be explained later.

#### Expected outputs
It is important to fully understand the expected format of algorithm output for each task. As you know, the input to algorithms for both tasks is the same and is a NumPy array of shape `(N, 256, 256, 3)` where `N` is the number of test images. The code snippet needed to load this data from the Grand-Challenge platform as a NumPy image is given in the template code. Also, we tried to include the code needed to convert algorithm outputs and save results to disk for each task.

 - Task 1 - Nuclear segmentation and classification: In this task participants are expected to process all the images in the input array and create another array with the shape of  `(N, 256, 256, 2)` where `N` is the number of test images and 2 channels are for predicted instance map and class map. This array needs to be converted and saved in the `/output/` directory as a `.mha` file. A code snippet to do so is provided in the template of `main.py`.

 - Task 2 - Prediction of cellular composition: In this task, participants are expected to generate a JSON file for each type of cell which contains a list of the number of cells of that type in each image. Assuming the algorithm calculates a list of `N` items where item `n` in that list contains a cellular composition of the image at index `n` in the input (`input_array[n]`), there is template code in `main.py` which extract the cellular composition of each cell type and save it with appropriate JSON format.

### 3- Creating the docker container
Once you have your algorithm ready according to the challenge template, you can use the `Dockerfile` template in the repository to containerize your algorithm. All the instructions you need are commented in the file, however, we explain the important parts of the files you need to change when creating your docker container:

- `Dockerfile`: This is the main file responsible for docker configuration. Please make sure not to change the code parts specified by `"! DO NOT MODIFY"` as these are preset configurations agreed to be used with the Grand-challenge platform. However, you might need to `"! USER SPECIFIC"` regions. For example, you need to include your desired base image and Linux packages to be installed at the top of the `Dockerfile`. It is recommended to install your desired python packages using the list `requirements.txt` list (explained below) instead of installing them one by one within the `Dockerfile`.
- `requirements.txt`: This file contains a list of all python libraries that are required to run your code. Please make sure that you include everything you need and this can be checked by testing your docker build locally (explained in the next section).
- `build.sh`: Helper bash script to generate a docker container based on the provided `Dockerfile` in the directory. Remember, in order to run this script you need to have a working installation of Docker on your system.

In summary, there are two easy steps to create your docker container when you have the algorithm ready: First, you need to modify the `Dockerfile` and `requirements.txt` according to your needs and then run the `build.sh` bash script in your terminal to create the container:
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
Note that `--memory` argument should be set based on your system specifications.

### 5- Exporting the docker container
Assuming that you have passed all the previous steps successfully, you should be able to easily export your docker image in a compressed format which is the requirement for the Grand-Challenge platform. This is done by calling `export.sh` bash script:
```bash
sudo ./export.sh
```
Note that you will need the `gzip` library installed if you want to successfully run this script. This step creates a file with the extension "tar.gz", which you can then upload to grand-challenge to submit your algorithm, which will be explained in the next section.

===========================================================================

## Submit Your Algorithm

Assuming you have a verified Grand-Challenge account and have already registered for the CoNIC challenge, you need to do two main steps to submit your algorithm to the challenge. First, you need to [upload the algorithm](#uplaod-your-algorithm) docker container to the Grand-Challenge platform. Then, you can make a [submit that algorithm](#submit-your-algorithm) to compete in any leaderboard or phases of the challenge. But before you proceed, make sure that you have read and understood the [challenge rules](https://conic-challenge.grand-challenge.org/Rules/).

### 1- Upload your algorithm
> **IMPORTANT:** It is crucial to know that you have to submit different algorithms for different tasks of the challenge. Even if you are using the same method for both tasks, you have to upload your algorithm twice because the Input and Output configurations for the two tasks are different.

In order to submit your algorithm, you first have to add it to the Grand-Challenge platform. To do so, you have to follow the following steps: First, navigate to the [algorithm submission webpage](https://grand-challenge.org/algorithms/) and click on the "+ Add new algorithm" botttom:

<p align="center">
  <img src="/doc/algorithm.jpg">
</p> 

Then you will be directed to the "Create Algorithm" page where you have to fill some necessay fileds, as described below (please pay special attention to the files **Inputs** and **Outputs**):

<p align="center">
  <img src="/doc/algorithm fields.JPG">
</p>

- Title: title of your algorithm to be shown on the leaderboard and your dashboard.
- Contact email: the email of the person responsible for this algorithm (This email will be listed as the contact email for the algorithm and will be visible to all users of Grand Challenge.)
- Display editors: Should the editors of this algorithm be listed on the information page. Preferably selected "Yes".
- Logo: Uploading a small logo image is mandatory by the grand-challenge platform. Try using an image that represents your team or algorithm.
- Viewer: This selects the viewer that might be used for showing algorithm results. Please select: "Viewer CIRRUS Core (Public)".
- **Inputs**: The input interface to the algorithm. This field determines what kind of input file the algorithm container expects to see in the input. For both tasks 1 and task 2 of the challenge set this to be **Generic Medical Image (Image)**
- **Outputs**: The output interfaces for the algorithm. This field specifies the type of output(s) that algorithm generates. Please note that the output types are different for tasks 1 and 2:
   1. **Task 1**: the "Outputs" field should be set to **Generic Medical Image (Image)**.
   2. **Task 2**: you should select 6 types of output interfaces, each representing cell count for a specific cell type. These interfaces are: **Epithelial Cell Count (Anything)**, **Lymphocyte Cell Count (Anything)**, **Plasma Cell Count (Anything)**, **Neutrophil Cell Count (Anything)**, **Eoinophil Cell Count (Anything)**, **Connective tissue Cell Count (Anything)**.

<p align="center">
<img src="/doc/task1_input_output.JPG">
</p>

<p align="center">
<img src="/doc/task2_input_output.JPG">
</p>

- Credits per job: keep this to 0.
- Image requires gpu: make sure to enable (check) the use of GPU if your algorithm needs one.
- Image requires memory gb: Specify how much RAM your algorithm requires to run. The maximum amount allowed is 32.

Once you have completed these required fields, press the "**Save**" button at the bottom of the page to create the algorithm and direct to the algorithm page where you can see the information regarding your algorithm and change them using "**Update Setting**" button if needed. Before you can use this algorithm for a challenge submission, you have to assign/upload your dockerized algorithm to it. To do so, click on the "**Containers**" tab from the left menu:

<p align="center">
<img src="/doc/algorithm_page.JPG">
</p>

Then, you have to click on the ![Upload Container](/doc/upload_container_botton.JPG) to navigate to the page where you can upload the packaged (compressed) docker container:

<p align="center">
<img src="/doc/container_upload.JPG">
</p>

Once you have uploaded your docker container and set the "GPU Supported" and "Requires memory gb" options (as explained before), click on the "Save" button and your algorithm will be completed and ready to be submitted to the challenge. Remember, the algorithm is only ready to submit when the status badge in front of the upload description changes to "Active".

### 2- Submit your algorithm
In the CoNIC challenge, we have two tasks (Task 1: cell segmentation and classification, Task 2: Cellular composition prediction) and for each task, participants compete in two phases (Preliminary test phase, Final test phase). The important phase in which participants are ranked for prizes is Phase 2 for both tasks, however, participants are encouraged to submit their algorithm to the preliminary phase as well to evaluate their method and check for the sanity of dockerized algorithms.
To start with your submission, for each task on either phases, you have to navigate to the challenge ["Submission" page](https://conic-challenge.grand-challenge.org/evaluation/challenge/submissions/create/):

<p align="center">
<img src="/doc/submissions.JPG">
</p>

on the top region, you can select for which phase and task you are submitting your method. Assuming that we want to test it on the preliminary test phase, we select the "**Segmentation and Classification - Preliminary Test**" tab.

<p align="center">
<img src="/doc/submit_algorithm.jpg">
</p>

The most important thing here is to select the algorithm you created for this task from the "**Algorithms**" list. You can also write comments about the submitted algorithm. Also, if you are submitting an algorithm for one of the tasks in the "Final Test Phase", it is mandatory to past a link to the ArXiv manuscript in which you have explained the technical details of your algorithm in the **Preprint (Algorithm Description)** field. Finally, by clicking on the "**Save**" button you will submit your algorithm for evaluation on the challenges task. The process is the same for both tasks and phases.
