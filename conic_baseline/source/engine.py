"""This module implements patch-level prediction."""

import copy
import imp
import os
import pathlib
import sys
import warnings
from collections import OrderedDict
from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.utils.data as torch_data
import tqdm

from .utils import mkdir, rm_n_mkdir


def model_to(on_gpu, model):
    """Transfers model to cpu/gpu.

    Args:
        on_gpu (bool): Transfers model to gpu if True otherwise to cpu
        model (torch.nn.Module): PyTorch defined model.

    Returns:
        model (torch.nn.Module):

    """
    if on_gpu:  # DataParallel work only for cuda
        model = torch.nn.DataParallel(model)
        return model.to("cuda")

    return model.to("cpu")


class FileLoader(torch.utils.data.Dataset):
    """A data loader.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """
    def __init__(
        self,
        img_path,
        indices=None,
        input_shape=None,
    ):
        self.imgs = np.load(img_path, mmap_mode='r')

        self.indices = (
            indices if indices is not None
            else np.arange(0, self.imgs.shape[0])
        )

        self.input_shape = input_shape
        return

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        # RGB images
        img = np.array(self.imgs[idx]).astype("uint8")
        return idx, img


class PatchPredictor:

    def __init__(
        self,
        batch_size=8,
        num_loader_workers=0,
        model=None,
        verbose=True,
    ):
        super().__init__()

        self.model = model  # for runtime, such as after wrapping with nn.DataParallel
        self.batch_size = batch_size
        self.num_loader_workers = num_loader_workers
        self.verbose = verbose

    def predict(self, dataset, save_dir, on_gpu=True):
        # if not os.path.exists(save_dir):
        rm_n_mkdir(save_dir)

        # use external for testing
        self._model = model_to(on_gpu, self.model)

        loader = torch_data.DataLoader(
            dataset,
            drop_last=False,
            batch_size=self.batch_size,
            num_workers=self.num_loader_workers,
        )
        self._loader = loader

        pbar_desc = "Process Batch: "
        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=int(len(self._loader)),
            ncols=80,
            ascii=True,
            position=0,
        )

        cum_output = []
        for _, batch_data in enumerate(self._loader):
            sample_infos, sample_datas = batch_data
            batch_size = sample_infos.shape[0]
            # ! depending on the protocol of the output within infer_batch
            # ! this may change, how to enforce/document/expose this in a
            # ! sensible way?

            # assume to return a list of L output,
            # each of shape N x etc. (N=batch size)
            sample_outputs = self.model.infer_batch(
                self._model,
                sample_datas,
                on_gpu,
            )
            # repackage so that its a N list, each contains
            # L x etc. output
            sample_outputs = [np.split(v, batch_size, axis=0) for v in sample_outputs]
            sample_outputs = list(zip(*sample_outputs))

            # tensor to numpy, costly?
            sample_infos = sample_infos.numpy()
            sample_infos = np.split(sample_infos, batch_size, axis=0)

            sample_outputs = list(zip(sample_infos, sample_outputs))
            cum_output.extend(sample_outputs)

            # TODO: detach or hook this into a parallel process
            self._process_predictions(cum_output, save_dir)
            pbar.update()
        pbar.close()

    def _process_predictions(self, cum_output, save_dir):
        for sample_info, sample_output in cum_output:
            sample_info = int(sample_info)  # weird packing due to pytorch
            for head_idx, head_output in enumerate(sample_output):
                head_output = head_output[0]  # remove the batch dimension
                np.save(f"{save_dir}/{sample_info}.{head_idx}.npy", head_output)
