<p align="center">
  <img src="/doc/conic_banner.png">
</p>

# Docker Template for CoNIC Challenge Submission

This repository contains instructions for creating a valid docker for [CoNIC challenge](https://conic-challenge.grand-challenge.org/Home/) submissions.

For the preface, submitting a docker container is a great advantage because participants have full control over their algorithm setup and privacy. It is also tremendously beneficial for the organizers because we do not need to spend time setting up each different configuration. However, it should be noted that for this form of submission to work, we demand a strict definition of the I/O between the Grand Challenge platform and the docker container. 

> **Important**: Any violation of the protocols mentioned below will automatically void your submission results.

# Table of Contents
- [Dockerize Your Algorithm](#dockerize)
  1. [Prerequisite](#requirements)
  2. [Docker Container and Grand Challenge API](#docker_io)
  3. [Creating and Testing the Docker Container](#test_docker)
  4. [Exporting the Docker Container](#export_docker)
- [Summary](#summary)

# Dockerize Your Algorithm <a name="dockerize"></a>

The following steps should be done to create a valid docker container for the challenge.

## 1. Prerequisite <a name="requirements"></a>
To create and test your docker setup, you will need to install [Docker Engine](https://docs.docker.com/engine/install/)
and [NVIDIA-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (in case you need GPU computation).

After installing the docker, you can start by either copying this [folder]() and its contents; or cloning this repository and navigate to `conic_template` folder:

```
git clone -b docker-template https://github.com/TissueImageAnalytics/CoNIC
```
> **Note:** If you are a Windows user and wish to dockerize your algorithm with Cuda capabilities, we recommend using WSL for dockerization. Details of doing this is available [here](https://github.com/TissueImageAnalytics/CoNIC/blob/docker-template/READEME_windows.md).

## 2. Docker Container and Grand Challenge API <a name="docker_io"></a>

The Grand Challenge platform will use the following entry within your docker to provide input and retrieve output:
- `/opt/algorithm/`: Folder within the docker container that contains participant algorithm and all associated external data.
- `/input/*`: Folder within the docker container that contains input data **from the organizers**. For this challenge, each task will received one single `*.mha` file.
- `/output/*`: Folder within the docker container that contains output data **from the participants**. For this challenge, it is further defined that.
  - **Task 1**: A single `*.mha` to contain the segmentation results. This `*.mha` is an `int32` array that is of shape `Nx256x256x2`. The channel `0` contains nuclei instance `id` while the channel `1` contains the type of the instance at the same location. Please refer to the training ground truth provided in the challenge as an example.
  - **Task 2**: The results for counting `neutrophil`,
  `epithelial`, `lymphocyte`, `plasma`, `eosinophil` and
  `connective` nuclei must be respectively saved at the following locations:
    - `/output/neutrophil-count.json`
    - `/output/epithelial-cell-count.json`
    - `/output/lymphocyte-count.json`
    - `/output/plasma-cell-count.json`
    - `/output/eosinophil-count.json`
    - `/output/connective-tissue-cell-count.json`

Each file contains a list of integers of length `N`.

Before continuing, we outline the conventions we use within the files that give instructions for the user

```
# Instruction / Directive
# <<<<<<<<<<<<<<<<<<<<<<<<<
# some codes within this
# >>>>>>>>>>>>>>>>>>>>>>>>>
```
The above snippet means that any code within the `<<<` and `>>>` follows the instruction above it. For example

```
# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>
```
means that users should modify the content in between as they see approriate. However,
```
# ! DO NOT MODIFY
# <<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>
```
means that the content in between must **not** be modified or overwritten at all costs!

Now, in line with the above API, we pre-define and hard-code the input and output conversion components of the code. Participants should avoid editing this if they are unclear about the API and how docker works.

- `Dockerfile`: Contains the instruction for [Docker Engine](https://docs.docker.com/engine/install/) so that they can build your docker container.

- `process.py`: This is the file that we hard code within `Dockerfile` as entry point. When Grand Challenge platform run your docker, this file will be executed and it will call the `run` from `main.py` with the appropriate entry point. We have designed this file and `​source/main.py` based on our agreement with the Grand Challenge about the docker I/O. For **debugging** your python code **locally outside docker**, you need to set `EXECUTE_IN_DOCKER = False` and change the `LOCAL_ENTRY` dictionary values (for `"input_dir"`, `"output_dir"`, and `user_data_dir` keys) according to your system.


- `source/main.py`: This file contains a `run` function that is called by the `process.py`. The I/O of this `run` function has been pre-defined based on our (the organizers) aggreement with the Grand-Challenge system so that they can provide input to your docker and pick up your docker predictions for evaluation. This `run` function is where your entire algorithm will be executed. We expect you to fill in the code to do so within

```
# ===== Whatever you need (function calls or complete algorithm) goes here
# <<<<<<<<<<<<<<<<<<<<<<<<<
# ...
# >>>>>>>>>>>>>>>>>>>>>>>>>
```
- `source/utils.py`: This file contains miscellaneous functions that are required by the portion we defined within `source/main.py` and `process.py`. Feel free to use any function within this for your purpose but please do not modify or remove it.

- `data` directory: All external data that users require to run their algorithms should be put in this directory. This data includes model trained weights, stain normalization
target, etc. We have designed the `Dockerfile` template so the contents in `data` are automatically copied over to your docker container under `/opt/algorithm/data/`.

> **Important**: Please note that your docker container will not have interent access when it's being run on the Grand Challenge platform. Therefore, all data and packages required for the algorithm should be included in the docker container upon its creation.

- `requirements.txt`: This file contains a list of all python libraries that are required to run your code. Please make sure that you include everything you need and this can be checked by testing your docker build locally (explained in the next section).

> **Important**: All libraries that have been defined within the current `requirements.txt` must not be removed for the template `source/main.py` and `process.py` to function normally.

- `build.sh`: Helper bash script to generate a docker container based on the provided `Dockerfile` in the directory. Remember, in order to run this script you need to have a working installation of Docker on your system.

## 3. Creating and Testing the Docker Container <a name="test_docker"></a>

Once you successfully test `source/main.py` locally for your algorithm, modify the `Dockerfile` and `requirements.txt` according to your needs and then run the `build.sh` bash script in your terminal to create the container:

```bash
sudo ./build.sh
```

Additionally, you can use the following script to create and test run your docker container on your local machine.
```bash
sudo ./test.sh
```

This bash script `test.sh` first tries to build the docker container by first calling `build.sh` internally and then running the docker container based on its defined entry point.

We have defined the entry point as following within the `Dockerfile` to run the aforementioned `process.py`

```
ENTRYPOINT python -m process $0 $@
```

To run `test.sh` sucessfully, you will need to modify the `LOCAL_INPUT` and `LOCAL_OUTPUT` variables to point them to the directories that respectively contain the input `*.mha` and the inference results. You can use the following snippet to convert `*.npy` images to `*.mha`
for testing.

```python
import itk
import numpy as np

arr = np.load(f"{DATA_ROOT_DIR}/images.npy")
dump_itk = itk.image_from_array(arr)
itk.imwrite(dump_itk, f"{OUT_DIR}/images.mha")
dump_np = itk.imread(f"{OUT_DIR}/images.mha")
dump_np = np.array(dump_np)
# content check
assert np.sum(np.abs(dump_np - arr)) == 0
```

## 4. Exporting the Docker Container <a name="export_docker"></a>
Assuming that you have successfully passed all of the previous steps, you need to export your docker container to a file that is fitted for submission. This is done by calling `export.sh` bash script:
```bash
sudo ./export.sh
```
Note that you will need the `gzip` library installed if you want to successfully run this script. This step creates a file with the extension "tar.gz", which you can then upload to Grand Challenge to submit your algorithm.

For submission guidelines, please refer to this [page](https://github.com/TissueImageAnalytics/CoNIC/tree/docker-template).

# Summary <a name="summary"></a>

Assuming you have understood the above, here we provide a short summary of the described steps:

1. Move your model weights (and other files if needed) into `data` folder.
2. Move your code into `source` folder.
3. Modify `main.py` to call your packaged code.
4. Modify `LOCAL_ENTRY` and set `EXECUTE_IN_DOCKER=False` within `process.py` for local debugging.
5. Use local python debugger to run and test `process.py`. This tests your code that has been added to `main.py`. Repeat previous steps upon failure.
6. Modify `Dockerfile` to dockerize your code if necessary. You will likely skip this step if you have not deviated from our instructions.
7. Set `EXECUTE_IN_DOCKER=True` within `process.py` and modify `requirements.txt` according to your needs.
8. Modify `LOCAL_INPUT` and `LOCAL_OUTPUT` within `./test.sh`.
9. Run `test.sh` for testing the docker container locally. Repeat previous steps upon failure.
10. Run `export.sh` for generating docker container for submission.
11. Make your Algorithm on Grand Challenge website. Please refer to this [page](https://github.com/TissueImageAnalytics/CoNIC/tree/docker-template) for setting up the interface.
12. Upload docker container from step #10 to your Grand Challenge Algorithm.
13. Submit your algorithm to CoNIC Challenge.
