<p align="center">
  <img src="/doc/conic_banner.png">
</p>

# Docker Template for CoNIC Challenge Submission

This repository contains instructions for creating a valid docker for [CoNIC challenge](https://conic-challenge.grand-challenge.org/Home/) submissions.

For the preface, submitting a docker image is a great advantage because participants have full control over their algorithm setup and privacy. It is also tremendously beneficial for the organizers because we do not need to spend time setting up each different configuration. However, it should be noted that for this form of submission to work, we demand a strict definition of the I/O between the Grand Challenge platform and the docker image. 

> **Important**: Any violation of the protocols mentioned below will automatically void your submission results.


# Dockerize Your Algorithm

The following steps should be done to create a valid docker image for the challenge.

## 1. Prerequisite
To create and test your docker setup, you will need to install [Docker Engine](https://docs.docker.com/engine/install/)
and [NVIDIA-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (in case you need GPU computation).

After installing the docker, you can start by either copying this [folder]() and its contents; or cloning this repository and navigate to `conic_template` folder:

```
git clone -b docker-template https://github.com/TissueImageAnalytics/CoNIC
```

## 2.Docker Image and Grand Challenge API

The Grand Challenge platform will use the following entry within your docker to provide input and retrieve output:
- `/opt/algorithm/`: Folder within the docker image that contains participant algorithm and all associated external data.
- `/input/*`: Folder within the docker image that contains input data *from the organizers*. For this challenge, each task will received one single `*.mha` file.
- `/output/*`: Folder within the docker image that contains output data *from the participants*. For this challenge, it is further defined that.
  - **Task 1**: A single `*.mha` to contain the segmentation results. This is `*.mha` is an `int32` array that is of shape `Nx256x256x2`. The channel `0` contains nuclei instance `id` while the channel `1` contains the type of the instance at the same location. Please refer to the training ground truth provided in the challenge as an example.
  - **Task 2**: The results for counting `neutrophil`,
  `epithelial`, `lymphocyte`, `plasma`, `eosinophil` and
  `connective` must be respectively saved at followings. Each file contain a list of integer of length `N`.
    - `/output/neutrophil-count.json`
    - `/output/epithelial-cell-count.json`
    - `/output/lymphocyte-count.json`
    - `/output/plasma-cell-count.json`
    - `/output/eosinophil-count.json`
    - `/output/connective-tissue-cell-count.json`

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

- `Dockerfile`: Contains the instruction for [Docker Engine](https://docs.docker.com/engine/install/) so that they can build your docker image.

- `process.py`: This is the main file that we hard code the `Dockerfile` to run on the Grand Challenge Platform to make it easy for you. For **debugging** your python code **locally outside docker**, you need to set `EXECUTE_IN_DOCKER = False` and change the `LOCAL_ENTRY` dictionary values (for `"input_dir"`, `"output_dir"`, and `user_data_dir` keys) according to your system.


- `source/main.py`: This file contains a `run` function that is called by the `process.py`. The I/O of this `run` function has been pre-defined based on our (the organizers) aggreement with the Grand-Challenge system so that they can provide and pick up your predictions for evaluation. This `run` function is where your entire algorithm will be executed. We expect you to fill in the code to do so within

```
# ===== Whatever you need (function calls or complete algorithm) goes here
# <<<<<<<<<<<<<<<<<<<<<<<<<
# ...
# >>>>>>>>>>>>>>>>>>>>>>>>>
```
- `source/utils.py`: This file contains miscellaneous functions that are required by the portion we defined within `source/main.py` and `process.py`. Feel free to use any function within this for your purpose but please do not modify or remove it.

- `data` directory: All external data that users require to run their algorithms should be put in this directory. This data includes model trained weights, stain normalization
target, etc. We have designed the `Dockerfile` template so the contents in `data` are automatically copied over to your docker image under `/opt/algorithm/data/`.

> **Important**: Please note that your docker container will not have interent access when it's being run on the Grand Challenge platform. Therefore, all data and packages required for the algorithm should be included in the docker container upon its creation.

- `requirements.txt`: This file contains a list of all python libraries that are required to run your code. Please make sure that you include everything you need and this can be checked by testing your docker build locally (explained in the next section).

> **Important**: All libraries that have been defined within the current `requirements.txt` must not be removed for the template `source/main.py` and `process.py` to function normally.

- `build.sh`: Helper bash script to generate a docker container based on the provided `Dockerfile` in the directory. Remember, in order to run this script you need to have a working installation of Docker on your system.

## 3. Creating and Testing the docker container

Once you successfully test `source/main.py` locally for your algorithm, modify the `Dockerfile` and `requirements.txt` according to your needs and then run the `build.sh` bash script in your terminal to create the container:

```bash
sudo ./build.sh
```

Additionally, you can use the following script to create and test run your docker image on your local machine.
```bash
sudo ./test.sh
```

This bash script `test.sh` first tries to build the docker container (if it hasn't been built yet) by first calling `build.sh` internally and then running the docker image based on its defined entry point.

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

## 4. Exporting the docker container
Assuming that you have successfully passed all of the previous steps, you need to export your docker image to a file that is fitted for submission. This is done by calling `export.sh` bash script:
```bash
sudo ./export.sh
```
Note that you will need the `gzip` library installed if you want to successfully run this script. This step creates a file with the extension "tar.gz", which you can then upload to Grand Challenge to submit your algorithm.

For submission guidelines, please refer to this [page]().
