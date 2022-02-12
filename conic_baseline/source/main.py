

import itk
import logging
import os
import warnings

warnings.filterwarnings("ignore")

import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

from .engine import FileLoader, PatchPredictor
from .net_desc import HoVerNetConic
from .utils import (
    cropping_center, overlay_prediction_contours,
    recur_find_ext, rm_n_mkdir, rmdir, print_dir,
    save_as_json)


def process_segmentation(np_map, hv_map, tp_map, model):
    # HoVerNet post-proc is coded at 0.25mpp so we resize
    np_map = cv2.resize(np_map, (0, 0), fx=2.0, fy=2.0)
    hv_map = cv2.resize(hv_map, (0, 0), fx=2.0, fy=2.0)
    tp_map = cv2.resize(
                    tp_map, (0, 0), fx=2.0, fy=2.0,
                    interpolation=cv2.INTER_NEAREST)

    inst_map = model._proc_np_hv(np_map[..., None], hv_map)
    inst_dict = model._get_instance_info(inst_map, tp_map[..., None])

    # Generating results match with the evaluation protocol
    type_map = np.zeros_like(inst_map)
    inst_type_colours = np.array([
        [v['type']] * 3 for v in inst_dict.values()
    ])
    type_map = overlay_prediction_contours(
        type_map, inst_dict,
        line_thickness=-1,
        inst_colours=inst_type_colours)

    pred_map = np.dstack([inst_map, type_map])
    # The result for evaluation is at 0.5mpp so we scale back
    pred_map = cv2.resize(
                    pred_map, (0, 0), fx=0.5, fy=0.5,
                    interpolation=cv2.INTER_NEAREST)
    return pred_map


def process_composition(pred_map, num_types):
    # Only consider the central 224x224 region,
    # as noted in the challenge description paper
    pred_map = cropping_center(pred_map, [224, 224])
    inst_map = pred_map[..., 0]
    type_map = pred_map[..., 1]
    # ignore 0-th index as it is 0 i.e background
    uid_list = np.unique(inst_map)[1:]

    if len(uid_list) < 1:
        type_freqs = np.zeros(num_types)
        return type_freqs
    uid_types = [
        np.unique(type_map[inst_map == uid])
        for uid in uid_list
    ]
    type_freqs_ = np.unique(uid_types, return_counts=True)
    # ! not all types exist within the same spatial location
    # ! so we have to create a placeholder and put them there
    type_freqs = np.zeros(num_types)
    type_freqs[type_freqs_[0]] = type_freqs_[1]
    return type_freqs


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
    # ===== Header script for user checking
    print(f"INPUT_DIR: {input_dir}")
    # recursively print out all subdirs and their contents
    print_dir(input_dir)
    print("USER_DATA_DIR: ", os.listdir(user_data_dir))
    # recursively print out all subdirs and their contents
    print_dir(user_data_dir)
    print(f"OUTPUT_DIR: {output_dir}")

    print(f"CUDA: {torch.cuda.is_available()}")
    for device in range(torch.cuda.device_count()):
        print(f"---Device {device}: {torch.cuda.get_device_name(0)}")

    paths = recur_find_ext(f"{input_dir}", [".mha"])
    assert len(paths) == 1, "There should only be one image package."
    IMG_PATH = paths[0]

    # convert from .mha to .npy
    images = np.array(itk.imread(IMG_PATH))
    np.save("images.npy", images)

    # ===== Whatever you need

    # The number of nuclei within the dataset/predictions.
    # For CoNIC, we have 6 (+1 for background) types in total.
    NUM_TYPES = 7
    # The path to the pretrained weights
    PRETRAINED = f'{user_data_dir}/hovernet-conic.pth'
    # The path to contain output and intermediate processing results
    OUT_DIR = f'{output_dir}/'
    DOCKER_OUT_DIR = "predictions/"
    print(PRETRAINED)

    start_time = time.time()

    pretrained = torch.load(PRETRAINED)
    model = HoVerNetConic(num_types=NUM_TYPES)
    model.load_state_dict(pretrained)

    dataset = FileLoader(f"images.npy")

    # Tile prediction
    predictor = PatchPredictor(
        model=model,
        num_loader_workers=4,
        batch_size=8,
    )

    logger = logging.getLogger()
    logger.disabled = True

    # capture all the printing to avoid cluttering the console
    predictor.predict(
        dataset,
        on_gpu=True,
        save_dir=f'{DOCKER_OUT_DIR}/raw/'
    )
    end_time = time.time()
    print("Infer time: ", end_time - start_time)

    images = np.load(f"images.npy", mmap_mode="r")
    num_images = images.shape[0]

    semantic_predictions = []
    composition_predictions = []
    # for input_file, output_root in tqdm(output_info):
    for idx in range(num_images):
        img = np.array(images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np_map = np.load(f"{DOCKER_OUT_DIR}/raw/{idx}.0.npy")
        hv_map = np.load(f"{DOCKER_OUT_DIR}/raw/{idx}.1.npy")
        tp_map = np.load(f"{DOCKER_OUT_DIR}/raw/{idx}.2.npy")

        pred_map = process_segmentation(
            np_map, hv_map, tp_map, model)
        type_freqs = process_composition(
            pred_map, NUM_TYPES)

        semantic_predictions.append(pred_map)
        composition_predictions.append(type_freqs)
    semantic_predictions = np.array(semantic_predictions)
    composition_predictions = np.array(composition_predictions)

    # ! >>>>>>>>>>>> Saving to approriate format for evaluation docker

    # Saving the results for segmentation in .mha
    itk.imwrite(
        itk.image_from_array(semantic_predictions),
        f"{OUT_DIR}/pred_seg.mha"
    )

    # version v0.0.8
    # Saving the results for composition prediction
    TYPE_NAMES = [
        "neutrophil",
        "epithelial-cell",
        "lymphocyte",
        "plasma-cell",
        "eosinophil",
        "connective-tissue-cell"
    ]
    for type_idx, type_name in enumerate(TYPE_NAMES):
        cell_counts = composition_predictions[:, (type_idx+1)]
        cell_counts = cell_counts.astype(np.int32).tolist()
        save_as_json(
            cell_counts,
            f'{OUT_DIR}/{type_name}-count.json'
        )

    TYPE_NAMES = [
        "neutrophil", "epithelial", "lymphocyte",
        "plasma", "eosinophil", "connective"
    ]
    df = pd.DataFrame(
        composition_predictions[:, 1:].astype(np.int32),
    )
    df.columns = TYPE_NAMES
    df.to_csv(f'{OUT_DIR}/pred_count.csv', index=False)

    end_time = time.time()
    print("Run time: ", end_time - start_time)

    # ! <<<<<<<<<<<<
