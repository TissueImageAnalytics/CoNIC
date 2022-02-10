

<p align="center">
  <img src="doc/conic_banner.png">
</p>

# CoNIC: Template for Inference Docker

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

# Docker

To test your docker setup, you will need to install [Docker](https://docs.docker.com/engine/install/)
and [NVIDIA-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Once you have verified they are installed correctly.
You can test your docker by doing:

```bash
sudo ./test.sh
```
# Sample

We have packaged HoVerNet code into a docker [here]() which you can use as an example.

