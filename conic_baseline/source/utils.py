
import colorsys
import copy
import json
import os
import pathlib
import random
import shutil
from typing import Tuple, Union

import cv2
import numpy as np


def print_dir(root_path):
    """Print out the entire directory content."""
    for root, subdirs, files in os.walk(root_path):
        print(f"-{root}")
        for subdir in subdirs:
            print(f"--D-{subdir}")
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"--F-{file_path}")


def save_as_json(data, save_path):
    """Save data to a json file.

    The function will deepcopy the `data` and then jsonify the content
    in place. Support data types for jsonify consist of `str`, `int`, `float`,
    `bool` and their np.ndarray respectively.

    Args:
        data (dict or list): Input data to save.
        save_path (str): Output to save the json of `input`.

    """
    shadow_data = copy.deepcopy(data)

    # make a copy of source input
    def walk_list(lst):
        """Recursive walk and jsonify in place."""
        for i, v in enumerate(lst):
            if isinstance(v, dict):
                walk_dict(v)
            elif isinstance(v, list):
                walk_list(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
                walk_list(v)
            elif isinstance(v, np.generic):
                v = v.item()
            elif v is not None and not isinstance(v, (int, float, str, bool)):
                raise ValueError(f"Value type `{type(v)}` `{v}` is not jsonified.")
            lst[i] = v

    def walk_dict(dct):
        """Recursive walk and jsonify in place."""
        for k, v in dct.items():
            if isinstance(v, dict):
                walk_dict(v)
            elif isinstance(v, list):
                walk_list(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
                walk_list(v)
            elif isinstance(v, np.generic):
                v = v.item()
            elif v is not None and not isinstance(v, (int, float, str, bool)):
                raise ValueError(f"Value type `{type(v)}` `{v}` is not jsonified.")
            if not isinstance(k, (int, float, str, bool)):
                raise ValueError(f"Key type `{type(k)}` `{k}` is not jsonified.")
            dct[k] = v

    if isinstance(shadow_data, dict):
        walk_dict(shadow_data)
    elif isinstance(shadow_data, list):
        walk_list(shadow_data)
    else:
        raise ValueError(f"`data` type {type(data)} is not [dict, list].")
    with open(save_path, "w") as handle:
        json.dump(shadow_data, handle)


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred (ndarray): the 2d array contain instances where each instances is marked
            by non-zero integer.
        by_size (bool): renaming such that larger nuclei have a smaller id (on-top).

    Returns:
        new_pred (ndarray): Array with continguous ordering of instances.

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def cropping_center(x, crop_shape, batch=False):
    """Crop an array at the centre with specified dimensions."""
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def mkdir(dir_path):
    """Make directory if it does not exist."""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def recur_find_ext(root_dir, ext_list):
    """Recursively find all files in directories end with the `ext` such as `ext='.png'`.

    Args:
        root_dir (str): Root directory to grab filepaths from.
        ext_list (list): File extensions to consider.

    Returns:
        file_path_list (list): sorted list of filepaths.
    """
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def rm_n_mkdir(dir_path):
    """Remove and then make a new directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def get_bounding_box(img):
    """Get the bounding box coordinates of a binary input- assumes a single object.

    Args:
        img: input binary image.

    Returns:
        bounding box coordinates

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return np.array([rmin, rmax, cmin, cmax])


def random_colors(num_colors, bright=True):
    """Generate a number of random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.

    Args:
        num_colors(int): Number of perceptively different colors to generate.
        bright(bool): To use bright color or not.

    Returns:
        List of (r, g, b) colors.

    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / num_colors, 1, brightness) for i in range(num_colors)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def overlay_prediction_contours(
    canvas: np.ndarray,
    inst_dict: dict,
    draw_dot: bool = False,
    type_colours: dict = None,
    inst_colours: Union[np.ndarray, Tuple[int]] = (255, 255, 0),
    line_thickness: int = 2,
):
    """Overlaying instance contours on image.

    Internally, colours from `type_colours` are prioritized over
    `inst_colours`. However, if `inst_colours` is `None` and `type_colours`
    is not provided, random colour is generated for each instance.

    Args:
        canvas (ndarray): Image to draw predictions on.
        inst_dict (dict): Dictionary of instances. It is expected to be
            in the following format:
            {instance_id: {type: int, contour: List[List[int]], centroid:List[float]}.
        draw_dot (bool): To draw a dot for each centroid or not.
        type_colours (dict): A dict of {type_id : (type_name, colour)},
            `type_id` is from 0-N and `colour` is a tuple of (R, G, B).
        inst_colours (tuple, np.ndarray): A colour to assign for all instances,
            or a list of colours to assigned for each instance in `inst_dict`. By
            default, all instances will have RGB colour `(255, 255, 0)`.
        line_thickness: line thickness of contours.

    Returns:
        (np.ndarray) The overlaid image.

    """
    overlay = np.copy((canvas))

    if inst_colours is None:
        inst_colours = random_colors(len(inst_dict))
        inst_colours = np.array(inst_colours) * 255
        inst_colours = inst_colours.astype(np.uint8)
    elif isinstance(inst_colours, tuple):
        inst_colours = np.array([inst_colours] * len(inst_dict))
    elif not isinstance(inst_colours, (np.ndarray, list)):
        raise ValueError(
            f"`inst_colours` must be np.ndarray or tuple: {type(inst_colours)}"
        )
    inst_colours = np.array(inst_colours)

    for idx, [_, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colours is not None:
            inst_colour = type_colours[inst_info["type"]][1]
        else:
            inst_colour = inst_colours[idx]
            inst_colour = [int(v) for v in inst_colour]

        cv2.drawContours(
            overlay, [np.array(inst_contour)], -1, inst_colour, line_thickness
        )

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay
