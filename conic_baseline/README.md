<p align="center">
  <img src="/doc/conic_banner.png">
</p>

# Baseline Docker Image

This repository contains the baseline algorithm for CoNIC challenge that is readied to be dockerized and submitted to the CoNIC challenge. In this `README`, you will learn about this repository structure and how `conic_template` is modified. The baseline in question is the `HoVer-Net` that has been provided [here]().

Before you start, you need to download the `HoVer-net` [pretrained weight](https://drive.google.com/file/d/1oVCD4_kOS-8Wu-eS5ZqzE30F9V3Id78d/view?usp=sharing) and put it under `data` folder.

Compared to the baseline `conic_template`, this repository keeps the following files intact:
- `process.py`
- `test.sh`
- `export.sh`
- `Dockerfile`

We have modified `source/main.py` and fill in the 

```
# ===== Whatever you need (function calls or complete algorithm) goes here
# <<<<<<<<<<<<<<<<<<<<<<<<<
# ...
# >>>>>>>>>>>>>>>>>>>>>>>>>
```

with the function calls necessary to obtain our inference output. Notice that we have added

```python
def process_segmentation(np_map, hv_map, tp_map, model):
  ...

def process_composition(pred_map, num_types):
  ...
```
to `source\main.py` but we do not modify the I/O of the `run` functions. Strictly speaking, we have modified

```python
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
            f'{output_dir}/{type_name}.json'
        )
    # >>>>>>>>>>>>>>>>>>>>>>>>>
```

to a more succinct form to save the output. Here is the new version

```python
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
```

Notice that we keep the same form of outputs as defined before within `conic_template`. Other than that, we have packaged all required files under the `source` folder.

In order to test the docker image, you can use the following snippet to convert `*.npy` images to `*.mha`
for testing.

```python
import itk
import numpy as np

arr = np.load(f"{DATA_ROOT_DIR}/images.npy")
dump_itk = itk.image_from_array(arr)
itk.imwrite(dump_itk, f"{OUT_DIR}/images.mha")
dump_np = itk.imread(f"{OUT_DIR}/images.mha")
dump_np = np.array(dump_np)
# content check
assert np.sum(np.abs(dump_np - arr)) == 0
```
