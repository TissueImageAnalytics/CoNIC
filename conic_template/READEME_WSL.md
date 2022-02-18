# Algorithm Dockerization for Windows Users
If you are developing your algorithm in Windows operating system and wish to use GPU in your dockerized algorithm, we strongly recommend using [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about) for doing so. It can be done by following the official guidelines provided by NVidia: [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). However, we post here a general tutorial based on our experience with WSL. For this tutorial, we are using the [CoNIC baseline algorithm](https://github.com/TissueImageAnalytics/CoNIC/tree/docker-template/conic_baseline) which has been tested before. So, assuming that you also have your algorithm ready for dockerization, you can follow the below steps to successfully containerize your algorithms with GPU capability.

## Setup WSL with Cuda support
Naturally, you first need to install WSL on your Windows. Check if your system has the requirements for it [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#wsl2-system-requirements) and then continue with the following steps.
### 1- Install NVIDIA Driver for GPU Support
Download and install the latest display driver for your GPU from [here](https://developer.nvidia.com/cuda/wsl).

**Note: This is the only driver you need to install. Do not install any Linux display driver in WSL.** Also, if you are already using the Cuda toolkit in your Windows, most probably you won't need to install the GPU driver.

### 2- Install WSL
Launch your preferred Windows Terminal / Command Prompt / Powershell and install WSL:
```
wsl.exe --install
```
Ensure you have the latest WSL kernel:
```
wsl.exe --update
```
By default, WSL2 comes installed with Ubuntu which is our recommended distribution for this tutorial. **Note After the installation has finished you will need to restart your computer to complete the Ubuntu installation.**
Finally, to run WSL simply call it in a windows terminal (cmd):
```
wsl.exe
```
### Install the Cuda toolkit
There are various versions of the Cuda toolkit available for download, make sure you choose one that is compatible with your GPU and the deep learning package that you use. For this tutorial, we choose to go with **Cuda Toolkit 11.3**. You may need to change the following installation link accordingly if you select another version.

First, in the WSL terminal change the working directory to an address that you have read/write permission in and make a `Cuda` directory in which you will download and save Cuda installation files:
```
cd /mnt/wsl
mkdir cuda
cd cuda
```
Then, run the following commands one by one to download and install Cuda Toolkit:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-wsl-ubuntu-11-3-local_11.3.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-3-local_11.3.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

## Setup Docker in WSL and dockerize your algorithm
After you have WSL ready, you need to install "Docker" and "NVIDIA Container Toolkit" and then you will be able to dockerize your algorithm with GPU capability.

### 1- Install Docker
Use the Docker installation script to install standard Docker-CE for your choice of WSL 2 Linux distribution.
```
curl https://get.docker.com | sh 
```

### 2- Install NVIDIA Container Toolkit
Run the following commands one by one:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2  
```
Open a separate WSL 2 window and start the Docker daemon again using the following commands to complete the installation:
```
sudo service docker stop
sudo service docker start
```
[Optional] In other to make sure you have successfully installed Docker with Cuda capabilities, you can run the following command which runs an N-body simulation CUDA sample. This example has already been containerized and available from NGC.
```
docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```
if everything goes well, you should see the following message on your screen:
<p align="center">
<img src="/doc/WSL_cuda.JPG">
</p>

### 3- Dockerize your algorithm
When you run WSL, your local drives will be mounted on to `/mnt/` path of the WSL. So, for example, if have your algorithm ready based on our instructions [here](https://github.com/TissueImageAnalytics/CoNIC/tree/windows-docker/conic_template) and put it in the `D:\my_algorithm`, you should be able to access that path in WSL and call the provided dockerization bash script `build.sh` like below:
```
cd /mnt/d/my_algorithm
sudo build.sh
```
similarly, you can call the `export.sh` script to export the compressed algorithm container, ready for submission to the challenge:
```
sudo export.sh
```
Needless to say that you can directly access the created compressed container in Windows by navigating to `D:\my_algorithm`.

Looking forward to seeing your submissions on the leaderboard, Good Luck.
