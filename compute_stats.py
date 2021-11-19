"""compute_stats.py. Calculates the statistical measurements for the CoNIC Challenge.

Usage:
	compute_stats.py [--mode=<str>] [--pred=<path>] [--true=<path>]
	compute_stats.py (-h | --help)
	compute_stats.py --version

Options:
	-h --help                   Show this string.
	--version                   Show version.
    --mode=<str>                Choose either `regression` or `seg_class`.
	--pred=<path>               Path to the results directory.
	--true=<path>               Path to the ground truth directory.
"""

from docopt import docopt
import numpy as np
import os
import glob
import pandas as pd
import scipy.io as sio
from tqdm.auto import tqdm

from metrics.stats_utils import get_pq, get_multi_pq_info, get_multi_r2


if __name__ == "__main__":
    args = docopt(__doc__, version="CoNIC-stats-v1.0")

    mode = args["--mode"]
    pred_path = args["--pred"]
    true_path = args["--true"]

    # do initial checks
    if mode not in ["regression", "seg_class"]:
        raise ValueError("`mode` must be either `regression` or `seg_class`")

    all_metrics = {}
    if mode == "seg_class":
        # initialise empty placeholder lists
        pq_list = []
        mpq_info_list = []
        # get the path of directory containing the results
        pred_target = os.path.join(pred_path)
        # get the path of directory containing the ground truth
        true_target = os.path.join(true_path)

        # get the results - assume results to be .mat
        list_preds = glob.glob(pred_target + "/*.mat")
        for pred_file in tqdm(list_preds):
            metrics_names = ["pq", "multi_pq"]
            basename = os.path.basename(pred_file)
            basename = basename.split(".")[0]

            # assuming that result and ground truth file types are the same
            true_file = true_target + "/" + basename + ".mat"

            # load the respective mat files
            pred = sio.loadmat(pred_file)
            true = sio.loadmat(true_file)

            # ===============================================================
            #! get the results and ground truth instance map
            pred_inst = pred["inst_map"]
            true_inst = true["inst_map"]
            pred_class = pred["inst_type"]
            true_class = true["class"]

            for idx, metric in enumerate(metrics_names):
                if metric == "pq":
                    pq = get_pq(true_inst, pred_inst)
                    pq = pq[0][2]
                    pq_list[idx].append(pq)
                elif metric == "multi_pq":
                    # get the pq stat info from single image
                    mpq_info_single = get_multi_pq_info(true, pred)
                    mpq_info = []
                    # aggregate the stat info per class
                    for single_class_pq in mpq_info_single:
                        tp = len(single_class_pq[0])
                        fp = len(single_class_pq[1])
                        fn = len(single_class_pq[2])
                        sum_iou = single_class_pq[3]
                        mpq_info.append([tp, fp, fn, sum_iou])
                    mpq_info_list.append(mpq_info)

        pq_metrics = np.array(pq_list)
        pq_metrics_avg = np.mean(pq_metrics, axis=-1)
        if "multi_pq" in metrics_names:
            mpq_info_metrics = np.array(mpq_info_list, dtype="float")
            # sum over all the images
            total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)

        for idx, metric in enumerate(metrics_names):
            if metric == "multi_pq":
                mpq_list = []
                # for each class, get the multiclass PQ
                for cat_idx in range(total_mpq_info_metrics.shape[0]):
                    total_tp = total_mpq_info_metrics[cat_idx][0]
                    total_fp = total_mpq_info_metrics[cat_idx][1]
                    total_fn = total_mpq_info_metrics[cat_idx][2]
                    total_sum_iou = total_mpq_info_metrics[cat_idx][3]

                    # get the F1-score i.e DQ
                    dq = total_tp / (
                        (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6
                    )
                    # get the SQ, no paired has 0 iou so not impact
                    sq = total_sum_iou / (total_tp + 1.0e-6)
                    mpq_list.append(dq * sq)
                mpq_metrics = np.array(mpq_list)
                all_metrics[metric] = [np.mean(mpq_metrics)]
            else:
                all_metrics[metric] = [pq_metrics_avg[idx]]
        # ----------------

    else:
        # first check to make sure ground truth and prediction is in csv format
        if not os.path.isfile(true_path) or not os.path.isfile(pred_path):
            raise ValueError("pred and true must be in csv format.")

        pred_format = pred.split(".")[-1]
        true_format = true_path(".")[-1]
        if pred_path != "csv" or true_format != "csv":
            raise ValueError("pred and true must be in csv format.")

        pred_csv = pd.read_csv(pred_path)
        true_csv = pd.read_csv(true_path)

        r2 = get_multi_r2(true_csv, pred_csv)

        all_metrics["r2"] = [r2]

    df = pd.DataFrame(all_metrics)
    df = df.to_string(index=False)
    print(df)
    print()
