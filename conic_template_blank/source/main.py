
import os, glob
import torch
import itk
import numpy as np

from .utils import recur_find_ext, save_as_json

def run(
        input_dir: str,
        output_dir: str,
        user_data_dir: str,
    ) -> None:
    """Entry function for automatic evaluation.

    This is the function which will be called by the organizer
    docker template to trigger evaluation run. All the data
    to be evaluated will be provided in "input_dir" while
    all the results that will be measured must be saved
    under "output_dir". Participant auxiliary data is provided
    under  "user_data_dir".

    input_dir (str): Path to the directory which contains input data.
    output_dir (str): Path to the directory which will contain output data.
    user_data_dir (str): Path to the directory which contains user data. This
        data include model weights, normalization matrix etc. .

    """
    # ! DO NOT MODIFY - Importing images from GC platform
    # <<<<<<<<<<<<<<<<<<<<<<<<<
    print(f"INPUT_DIR: {input_dir}")
    print(f"OUTPUT_DIR: {output_dir}")
    print(f"CUDA: {torch.cuda.is_available()}")
    for device in range(torch.cuda.device_count()):
        print(f"---Device {device}: {torch.cuda.get_device_name(0)}")
    print("USER_DATA_DIR: ", os.listdir(user_data_dir))

    paths = recur_find_ext(f"{input_dir}", [".mha"])
    assert len(paths) == 1, "There should only be one image package."
    IMG_PATH = paths[0]

    # convert from .mha to .npy
    images = np.array(itk.imread(IMG_PATH)) # these are the images to work with

    # >>>>>>>>>>>>>>>>>>>>>>>>>


    # ===== Whatever you need (function calls or complete algorithm) goes here
    # <<<<<<<<<<<<<<<<<<<<<<<<<
    # ...
    # >>>>>>>>>>>>>>>>>>>>>>>>>


    # ! IMPORTANT: Template for creating the outputs for task 1 and 2
    # Saving the results for segmentation in .mha format
    # Expected `semantic_predictions` prediction array has the shape of
    # (N, 256, 256, 2)
    itk.imwrite(
        itk.image_from_array(semantic_predictions),
        f"{output_dir}/pred_seg.mha"
    )

    # version v0.0.3
    # Saving the results for composition prediction
    # Expected `composition_predictions` is a list of N entries in which
    # number of different types of cells is saved.
    composition_predictions = np.array(composition_predictions)
    TYPE_NAMES = [
        "neutrophil", "epithelial", "lymphocyte",
        "plasma", "eosinophil", "connective"
    ]
    for type_idx, type_name in enumerate(TYPE_NAMES):
        cell_counts = composition_predictions[:, (type_idx+1)]
        cell_counts = cell_counts.astype(np.int32).tolist()
        save_as_json(
            cell_counts,
            f'{output_dir}/{type_name}.json'
        )
