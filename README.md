<p align="center">
  <img src="conic_banner.png">
</p>

# CoNIC: Colon Nuclei Identification and Counting Challenge

In this repository we provide code and example notebooks to assist participants start their algorithm development for the CoNIC challenge. In particular we provide:

- Example notebooks
  - Data reading and statistics
- Evaluation code
  - Segmentation & classification: multi-class panoptic quality
  - Predicting cellular composition: multi-class coefficient of determination ($R^2$)


## Output format for metric calculation

To appropriately calculate the metrics, ensure that your output is in the following format:

- Segmentation and classification:

  - Single `.mat` file per input image, where the filename indicates the original image. For example, if we processed an image named `img1.png`, then the corresponding output will be named `img1.mat`. The `.mat` file should have the following keys:
    - `'instance_map'`: 2-dimensional array of size 256x256 and type int32 with pixel values ranging from 0 (background) to N (number of nuclei)
    - `'category'`: a 2-dimensional array of size Nx2 of type int32. The first column denotes indicates the id within the instance map and the second column gives the nuclei category prediction (ranging from 1 to 6).
  
- Composition prediction:
  - Single `.csv` file where the column headers should be:
    - `filename`
    - `epithelial`
    - `lymphocyte`
    - `plasma`
    - `eosinophil`
    - `neutrophil`
    - `connective`
  - Each entry in the csv file will determine the predicted count for a given input image and category.


## Metric calculation
  To get the stats for segmentation and classification, run:

  ```
  python compute_stats.py --mode=seg_class --results=<path_to_results> --ground_truth=<path_to_ground_truth>
  ```
  
  To get the stats for cellular composition prediction, run:

  ```
  python compute_stats.py --mode=regression --results=<path_to_results> --ground_truth=<path_to_ground_truth>
  ```

  

