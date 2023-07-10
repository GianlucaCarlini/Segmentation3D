import numpy as np
import torch
from typing import Union
from utils.window_operations import get_patch_coords, pad_3d_array
from time import time


def predict_tensor_patches(
    tensor: torch.Tensor,
    model: torch.nn.Module,
    patch_size: tuple,
    strides: tuple,
    padding: str = "valid",
    unpad: bool = False,
    verbose: bool = False,
):
    """Predict a volume array by patching it in a sliding window fashion.

    Args:
        tensor (torch.Tensor): The input tensor to predict.
        model (torch.Module): The model to use for prediction.
        patch_size (tuple): Patch dimensions as a tuple of ints representing the z, y, and x
            dimensions. If a single int is provided, the same value will be used for z, y, and x.
        strides (tuple): Stride dimensions as a tuple of ints representing the z, y, and x
            strides. If a single int is provided, the same value will be used for z, y, and x.
        padding (str, optional): The type of padding to use. Can be "same" or "valid".
            if "same", the array is zero padded so that all the possible patches can be extracted.
            if "valid", the array is not padded and only patches that fully fit in the array are
            extracted. Defaults to "valid".
        unpad (bool, optional): Whether to unpad the array after prediction. Only used if padding
            is set to "same". Defaults to True.
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        tensor_pred (torch.Tensor): The predicted tensor.
    """

    model.eval()
    classes = model.classes

    tensor, pad_values = pad_3d_array(
        tensor, patch_size, strides, padding, return_pad=True
    )

    if unpad and padding == "valid":
        Warning(
            "no padding was used, unpadding will have no effect, setting unpad to False"
        )
        unpad = False

    coords_top_corner, coords_bottom_corner = get_patch_coords(
        tensor,
        patch_size,
        strides,
    )

    n_patches = coords_top_corner.shape[0]

    tensor_pred = torch.zeros(size=(classes, *tensor.shape)).to(tensor)
    mask = torch.zeros(size=(classes, *tensor.shape)).to(tensor)

    n = 0
    tic = time()

    for i in range(n_patches):
        top_corner = coords_top_corner[i]
        bottom_corner = coords_bottom_corner[i]

        patch = tensor[
            top_corner[0] : bottom_corner[0],
            top_corner[1] : bottom_corner[1],
            top_corner[2] : bottom_corner[2],
        ]

        mask[
            :,
            top_corner[0] : bottom_corner[0],
            top_corner[1] : bottom_corner[1],
            top_corner[2] : bottom_corner[2],
        ] += 1

        # if torch.eq(torch.count_nonzero(patch), torch.tensor(0).to(tensor)):
        #     n += 1
        #     if verbose:
        #         print(f"predicted {n}/{n_patches} patches \r", end="", flush=True)
        #     continue

        patch = patch.unsqueeze(0).unsqueeze(0)
        patch_pred = model(patch)
        patch_pred = patch_pred.squeeze(0)

        tensor_pred[
            :,
            top_corner[0] : bottom_corner[0],
            top_corner[1] : bottom_corner[1],
            top_corner[2] : bottom_corner[2],
        ] += patch_pred

        if verbose:
            n += 1
            print(f"predicted {n}/{n_patches} patches \r", end="", flush=True)

    if verbose:
        toc = time()
        print(f"prediction took {toc-tic:.2f} seconds")

    tensor_pred = tensor_pred / mask

    if unpad:
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = pad_values

        tensor_pred = tensor_pred[
            :,
            pad_front:-pad_back,
            pad_top:-pad_bottom,
            pad_left:-pad_right,
        ]
        return tensor_pred

    return tensor_pred
