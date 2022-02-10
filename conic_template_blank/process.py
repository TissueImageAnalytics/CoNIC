import os
from source.main import run

# Define the entry point for hooking data

# TODO: ask if the mounting entry point in docker is fixed from grand challenge team

# ! DO NOT MODIFY - ORGANIZER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
DOCKER_ENTRY = {
    "input_dir": "/input/",
    "output_dir": "/output/",
    "user_data_dir": "/opt/algorithm/data/"
}
# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
LOCAL_ENTRY = {
    "input_dir": "/mnt/storage_0/workspace/nuclei/conic-challenge/exp_output/local/data/valid/",
    "output_dir": "/mnt/storage_0/workspace/nuclei/conic-challenge/exp_output/dump/",
    "user_data_dir": "/mnt/storage_0/workspace/nuclei/conic-challenge/docker/inference/data/"
}
# >>>>>>>>>>>>>>>>>>>>>>>>>

# We have this parameter to adapt the paths between local execution
# and execution in docker. You can use this flag to switch between these two modes.

EXECUTE_IN_DOCKER = False


if __name__ == "__main__":
    print(f"\nWorking Directory: {os.getcwd()}")
    print("\n>>>>>>>>>>>>>>>>> Start User Script\n")
    # Trigger the inference in the `source` directory
    ENTRY = DOCKER_ENTRY if EXECUTE_IN_DOCKER else LOCAL_ENTRY
    run(**ENTRY)
    print("\n>>>>>>>>>>>>>>>>> End User Script\n")
