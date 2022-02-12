
This repository contains instructions and examples for creating a valid docker for [CoNIC challenge](https://conic-challenge.grand-challenge.org/Home/) and how you can submit it to the [CoNIC challenge](https://conic-challenge.grand-challenge.org/Home/) for evaluation.

## Table of Contents
1. [Creating Docker Image](#creating_docker)
2. [Submitting Docker Image](#submitting_docker)
3. [Video Tutorial](#video_tutorial)


## Creating Docker Image <a name="creating_docker"></a>

In this repository, you can find a template for creating valid a docker image for [CoNIC challenge](https://conic-challenge.grand-challenge.org/Home/). We also provide one example algorithm that has been prepared based on the aforementioned template:

- `conic_template`: This directory contains a template with all the essential functions and modules needed to create an acceptable docker container for submitting to the CoNIC challenge. Almost all of the functions and instructions in this template should remain the same and you just need to add/link your algorithm and weight files to them.
- `conic_baseline`: This directory contains a sample algorithm that has been prepared based on the aforementioned instructions within `conic_template`. For this example, the algorithm is the [CoNIC baseline method](https://github.com/vqdang/hover_net/tree/conic).

Each of these directories is accompanied by a `README.md` file in which we have thoroughly explained how you can dockerize your algorithms and submit them to the challenge. The code in the `conic_template` has been extensively commented and users should be able to embed their algorithms in the blank template, however, `conic_template_baseline` can be a good guide (example) to better understand the acceptable algorithm layout. 

## Submitting Docker Image <a name="submitting_docker"></a>


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

## Video Tutorial <a name="video_tutorial"></a>
For more information, please have a look at our [tutorial videos](https://conic-challenge.grand-challenge.org/Videos/).
