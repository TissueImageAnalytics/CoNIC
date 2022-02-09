"""extract_patches.py

Patch extraction script.
"""

import pathlib
import scipy.io as sio
import cv2
import numpy as np
import pandas as pd
import tqdm

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir, cropping_center, recur_find_ext, remap_label

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    win_size = 256  # should keep this the same!
    step_size = 256  # decrease this to have a larger overlap between patches
    extract_type = "valid"
    img_dir = "/Users/simongraham/Desktop/data/MTL_Data/Nuclei/Images/"
    ann_dir = "/Users/simongraham/Desktop/data/MTL_Data/Nuclei/Labels/"
    out_dir = "patches/"

    rm_n_mkdir(out_dir)

    xtractor = PatchExtractor(win_size, step_size)

    file_path_list = recur_find_ext(img_dir, ".png")

    pbar_format = (
        "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    )
    pbar = tqdm.tqdm(
        total=len(file_path_list), bar_format=pbar_format, ascii=True, position=0
    )

    img_list = []
    inst_map_list = []
    class_map_list = []
    nuclei_counts_list = []
    patch_names_list = []
    for file_idx, file_path in enumerate(file_path_list):
        basename = pathlib.Path(file_path).stem

        img = cv2.imread(img_dir + basename + ".png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ##
        ann_load = sio.loadmat(ann_dir + basename + ".mat")
        ann_inst = ann_load["inst_map"]
        inst_class = np.squeeze(ann_load["class"]).tolist()
        inst_id = np.squeeze(ann_load["id"]).tolist()
        ann_class = np.full(ann_inst.shape[:2], 0, dtype=np.uint8)
        for val in inst_id:
            ann_inst_tmp = ann_inst == val
            idx_tmp = inst_id.index(val)
            ann_class[ann_inst_tmp] = inst_class[idx_tmp]

        ann = np.dstack([ann_inst, ann_class])

        img = np.concatenate([img, ann], axis=-1)
        sub_patches = xtractor.extract(img, extract_type)

        for idx, patch in enumerate(sub_patches):
            patch_img = patch[..., :3]  # RGB image
            patch_inst = patch[..., 3]  # instance map
            patch_class = patch[..., 4]  # class map

            # ensure nuclei range from 0 to N (N is the number of nuclei in the patch)
            patch_inst = remap_label(patch_inst)

            # only consider nuclei for counting if it exists within the central 224x224 region
            patch_inst_crop = cropping_center(patch_inst, [224, 224])
            patch_class_crop = cropping_center(patch_class, [224, 224])
            nuclei_counts_perclass = []
            # get the counts per class
            for nuc_val in range(1, 7):
                patch_class_crop_tmp = patch_class_crop == nuc_val
                patch_inst_crop_tmp = patch_inst_crop * patch_class_crop_tmp
                nr_nuclei = len(np.unique(patch_inst_crop_tmp).tolist()[1:])
                nuclei_counts_perclass.append(nr_nuclei)

            img_list.append(patch_img)
            inst_map_list.append(patch_inst)
            class_map_list.append(patch_class)
            nuclei_counts_list.append(nuclei_counts_perclass)
            patch_names_list.append("%s-%04d" % (basename, idx))

            assert patch.shape[0] == win_size
            assert patch.shape[1] == win_size

        pbar.update()
    pbar.close()

    # convert to numpy array
    img_array = np.array(img_list).astype("uint8")
    inst_map_array = np.array(inst_map_list).astype("uint16")
    class_map_array = np.array(class_map_list).astype("uint16")
    nuclei_counts_array = np.array(nuclei_counts_list).astype("uint16")

    # combine instance map and classification map to form single array
    inst_map_array = np.expand_dims(inst_map_array, -1)
    class_map_array = np.expand_dims(class_map_array, -1)
    labels_array = np.concatenate((inst_map_array, class_map_array), axis=-1)

    # convert to pandas dataframe
    nuclei_counts_df = pd.DataFrame(
        data={
            "neutrophil": nuclei_counts_array[:, 0],
            "epithelial": nuclei_counts_array[:, 1],
            "lymphocyte": nuclei_counts_array[:, 2],
            "plasma": nuclei_counts_array[:, 3],
            "eosinophil": nuclei_counts_array[:, 4],
            "connective": nuclei_counts_array[:, 5],
        }
    )
    patch_names_df = pd.DataFrame(data={"patch_info": patch_names_list})

    # save output
    np.save(out_dir + "images.npy", img_array)
    np.save(out_dir + "labels.npy", labels_array)
    nuclei_counts_df.to_csv(out_dir + "counts.csv", index=False)
    patch_names_df.to_csv(out_dir + "patch_info.csv", index=False)

