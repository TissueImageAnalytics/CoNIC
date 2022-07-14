import os
from source.main import run

# Define the entry point for hooking data

# ! DO NOT MODIFY - ORGANIZER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
DOCKER_ENTRY = {
    # path to folder that contain .mha for inference
    "input_dir": "/input/",
    # path to folder that contain inference results,
    # select approriate output based on the comments below

    # ! This is for cellular composition submission
    "output_dir": "/output/",  
    # ! This is for segmentation and classification, the nested folder needed to be created!
    "output_dir": "/output/images/nuclear-segmentation-and-classification/",

    # path to folder that contain user data such as
    # neural network weights and stain matrices
    "user_data_dir": "/opt/algorithm/data/"
}
# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
LOCAL_ENTRY = {
    # path to folder that contain .mha for inference
    "input_dir": "",
    # path to folder that contain inference results
    "output_dir": "",
    # path to folder that contain user data such as
    # neural network weights and stain matrices
    "user_data_dir": ""
}
# >>>>>>>>>>>>>>>>>>>>>>>>>

# We have this parameter to adapt the paths between local execution
# and execution in docker. You can use this flag to switch between
# these two modes for debugging the docker image or your python code.

EXECUTE_IN_DOCKER = True

# ! DO NOT MODIFY - ORGANIZER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
if __name__ == "__main__":
    print(f"\nWorking Directory: {os.getcwd()}")
    print("\n>>>>>>>>>>>>>>>>> Start User Script\n")
    # Trigger the inference in the `source` directory
    ENTRY = DOCKER_ENTRY if EXECUTE_IN_DOCKER else LOCAL_ENTRY
    run(**ENTRY)
    print("\n>>>>>>>>>>>>>>>>> End User Script\n")
# >>>>>>>>>>>>>>>>>>>>>>>>>
