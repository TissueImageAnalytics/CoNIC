
import itk
import numpy as np

from .utils import print_dir, recur_find_ext, save_as_json


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
    # ! DO NOT MODIFY IF YOU ARE UNCLEAR ABOUT API !
    # <<<<<<<<<<<<<<<<<<<<<<<<< 
    # ===== Header script for user checking
    print(f"INPUT_DIR: {input_dir}")
    # recursively print out all subdirs and their contents
    print_dir(input_dir)
    print("USER_DATA_DIR: ")
    # recursively print out all subdirs and their contents
    print_dir(user_data_dir)
    print(f"OUTPUT_DIR: {output_dir}")

    paths = recur_find_ext(f"{input_dir}", [".mha"])
    assert len(paths) == 1, "There should only be one image package."
    IMG_PATH = paths[0]

    # convert from .mha to .npy
    images = np.array(itk.imread(IMG_PATH))
    np.save("images.npy", images)
    # >>>>>>>>>>>>>>>>>>>>>>>>>

    # ===== Whatever you need (function calls or complete algorithm) goes here
    # <<<<<<<<<<<<<<<<<<<<<<<<<
    # ...
    # >>>>>>>>>>>>>>>>>>>>>>>>>

    # ===== Modify this accordingly
    # <<<<<<<<<<<<<<<<<<<<<<<<<
    # ! Example of valid predictions
    num_images = 16
    np.random.seed(5)
    pred_segmentation = np.random.randint(
        0, 255, (num_images, 256, 256, 2), dtype=np.int32
    )
    pred_regression = {
        "neutrophil"            : np.random.randint(0, 255, num_images).tolist(),
        "epithelial-cell"       : np.random.randint(0, 255, num_images).tolist(),
        "lymphocyte"            : np.random.randint(0, 255, num_images).tolist(),
        "plasma-cell"           : np.random.randint(0, 255, num_images).tolist(),
        "eosinophil"            : np.random.randint(0, 255, num_images).tolist(),
        "connective-tissue-cell": np.random.randint(0, 255, num_images).tolist(),
    }
    # >>>>>>>>>>>>>>>>>>>>>>>>>

    # ! DO NOT MODIFY IF YOU ARE UNCLEAR ABOUT API !
    # <<<<<<<<<<<<<<<<<<<<<<<<<

    # For segmentation, the result must be saved at
    #     - /output/<uid>.mha
    # with <uid> is can anything. However, there must be
    # only one .mha under /output.
    itk.imwrite(
        itk.image_from_array(pred_segmentation),
        f"{output_dir}/pred_seg.mha"
    )

    # For regression, the result for counting "neutrophil",
    # "epithelial", "lymphocyte", "plasma", "eosinophil",
    # "connective" must be respectively saved at
    #     - /output/neutrophil-count.json
    #     - /output/epithelial-cell-count.json
    #     - /output/lymphocyte-count.json
    #     - /output/plasma-cell-count.json
    #     - /output/eosinophil-count.json
    #     - /output/connective-tissue-cell-count.json
    TYPE_NAMES = [
        "neutrophil",
        "epithelial-cell",
        "lymphocyte",
        "plasma-cell",
        "eosinophil",
        "connective-tissue-cell"
    ]
    for _, type_name in enumerate(TYPE_NAMES):
        cell_counts = pred_regression[type_name]
        save_as_json(
            cell_counts,
            f'{output_dir}/{type_name}-count.json'
        )
    # >>>>>>>>>>>>>>>>>>>>>>>>>
