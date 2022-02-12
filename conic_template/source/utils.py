import copy
import json
import os
import pathlib
import shutil
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
    `bool` and their `np.ndarray` respectively.

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

