import numpy as np
import torch
import torch.nn.functional as F
from typing import Union


def pad_3d_array(
    arr: Union[np.ndarray, torch.Tensor],
    patch_size: tuple,
    strides: tuple,
    padding: str = "valid",
    return_pad: bool = False,
) -> np.ndarray:
    """Pad a 3D array to make it compatible with the patch size and strides.
    The padding is performed in a tensorflow-like fashion.

    Args:
        arr (np.ndarray): The input array to pad
        patch_size (tuple): Patch dimensions as a tuple of ints representing the z, y, and x
            dimensions. If a single int is provided, the same value will be used for z, y, and x.
        strides (tuple): Stride dimensions as a tuple of ints representing the z, y, and x
            strides. If a single int is provided, the same value will be used for z, y, and x.
        padding (str, optional): The type of padding to use. Can be "same" or "valid".
            if "same", the array is zero padded so that all the possible patches can be extracted.
            if "valid", the array is not padded and only patches that fully fit in the array are
            extracted. Defaults to "valid".
        return_pad (bool, optional): Whether to return the padding values. Defaults to False.

    Raises:
        ValueError: If the input array is not 3D
        NotImplementedError: If the padding type is not "same" or "valid"

    Returns:
        arr (np.ndarray): The padded array

    Example:
        >>> arr = np.zeros((512, 512, 512))
        >>> arr = pad_3d_array(arr, patch_size=(128, 128, 128), strides=(64, 64, 64), padding="same")
        >>> arr.shape
        (576, 576, 576)
        >>> tensor = torch.zeros((1, 1, 512, 512, 512))
        >>> tensor = tensor.squeeze().squeeze()
        >>> tensor = pad_3d_array(tensor, patch_size=(128, 128, 128), strides=(64, 64, 64), padding="same")
        >>> tuple(tensor.shape)
        (576, 576, 576)

    """
    if arr.ndim != 3:
        raise ValueError("Input array must be 3D")

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)

    if isinstance(strides, int):
        strides = (strides, strides, strides)

    # I prefer to reason in terms of h, w, d instead of y, x, z
    d, h, w = tuple(arr.shape)

    if padding == "same":
        out_depth = np.ceil(d / strides[0]).astype(np.uint8)
        out_height = np.ceil(h / strides[1]).astype(np.uint8)
        out_width = np.ceil(w / strides[2]).astype(np.uint8)

        if d % strides[0] == 0:
            pad_along_depth = max(patch_size[0] - strides[0], 0)
        else:
            pad_along_depth = max(patch_size[0] - (d % strides[0]), 0)

        if h % strides[1] == 0:
            pad_along_height = max(patch_size[1] - strides[1], 0)
        else:
            pad_along_height = max(patch_size[1] - (h % strides[1]), 0)

        if w % strides[2] == 0:
            pad_along_width = max(patch_size[2] - strides[2], 0)
        else:
            pad_along_width = max(patch_size[2] - (w % strides[2]), 0)

        pad_front = pad_along_depth // 2
        pad_back = pad_along_depth - pad_front

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top

        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        if isinstance(arr, torch.Tensor):
            arr = F.pad(
                arr,
                pad=(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back),
                mode="constant",
                value=arr.min(),
            )
        elif isinstance(arr, np.ndarray):
            arr = np.pad(
                arr,
                pad_width=(
                    (pad_front, pad_back),
                    (pad_top, pad_bottom),
                    (pad_left, pad_right),
                ),
                mode="constant",
                constant_values=arr.min(),
            )
        else:
            raise TypeError(
                "Input array must be either a numpy.ndarray or a torch.Tensor"
            )

    elif padding == "valid":
        out_depth = np.ceil((d - patch_size[0] + 1) / strides[0]).astype(np.uint8)
        out_height = np.ceil((h - patch_size[1] + 1) / strides[1]).astype(np.uint8)
        out_width = np.ceil((w - patch_size[2] + 1) / strides[2]).astype(np.uint8)

        arr = arr[
            : patch_size[0] + out_depth * strides[0],
            : patch_size[1] + out_height * strides[1],
            : patch_size[2] + out_width * strides[2],
        ]

    else:
        raise NotImplementedError(
            "Padding type not implemented. Possible values are: 'same', 'valid'"
        )

    if return_pad and padding == "same":
        return arr, (pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right)

    return arr


def get_patch_coords(
    arr: Union[np.ndarray, torch.Tensor], patch_size: tuple, strides: tuple
) -> tuple:
    """Get the coordinates of the patches that can be extracted from an array.

    Args:
        arr (np.ndarray): The input array to patch.
        patch_size (tuple): Patch dimensions as a tuple of ints representing the z, y, and x
            dimensions. If a single int is provided, the same value will be used for z, y, and x.
        strides (tuple): Stride dimensions as a tuple of ints representing the z, y, and x
            strides. If a single int is provided, the same value will be used for z, y, and x.

    Returns:
        (coords_top_corner, coords_bottom_corner): Arrays of shape (n_patches, 3) containing the
            z, y, x coordinates of the top left and bottom right corners of each patch.

    Example:
        >>> arr = np.zeros((512, 512, 512))
        >>> arr = pad_3d_array(arr, patch_size=(128, 128, 128), strides=(64, 64, 64), padding="same")
        >>> coords_top_corner, coords_bottom_corner = get_patch_coords(arr, patch_size=(128, 128, 128), strides=(64, 64, 64))
        >>> coords_top_corner.shape
        (512, 3)
        >>> coords_bottom_corner.shape
        (512, 3)
        >>> tensor = torch.zeros((1, 1, 512, 512, 512))
        >>> tensor = tensor.squeeze().squeeze()
        >>> tensor = pad_3d_array(tensor, patch_size=(128, 128, 128), strides=(64, 64, 64), padding="same")
        >>> coords_top_corner, coords_bottom_corner = get_patch_coords(tensor, patch_size=(128, 128, 128), strides=(64, 64, 64))
        >>> coords_top_corner.shape
        (512, 3)
    """

    d, h, w = arr.shape

    n_patches_d = int((d - patch_size[0]) / strides[0]) + 1
    n_patches_h = int((h - patch_size[1]) / strides[1]) + 1
    n_patches_w = int((w - patch_size[2]) / strides[2]) + 1

    d_range = np.arange(0, n_patches_d, 1)
    h_range = np.arange(0, n_patches_h, 1)
    w_range = np.arange(0, n_patches_w, 1)

    i, j, k = np.meshgrid(d_range, h_range, w_range, indexing="ij")

    initial_i = i.ravel() * strides[0]
    initial_j = j.ravel() * strides[1]
    initial_k = k.ravel() * strides[2]

    final_i = initial_i + patch_size[0]
    final_j = initial_j + patch_size[0]
    final_k = initial_k + patch_size[0]

    coords_top_corner = np.stack((initial_i, initial_j, initial_k), axis=1)
    coords_bottom_corner = np.stack((final_i, final_j, final_k), axis=1)

    return (coords_top_corner, coords_bottom_corner)


def predict_array_patches(
    arr: Union[np.ndarray, torch.Tensor],
    model: torch.nn.Module,
    patch_size: tuple,
    strides: tuple,
    padding: str = "valid",
    device: str = "cpu",
    unpad: bool = True,
):
    """Predict a volume array by patching it in a sliding window fashion.

    Args:
        arr (np.ndarray): The input array to predict.
        model (torch.Module): The model to use for prediction.
        patch_size (tuple): Patch dimensions as a tuple of ints representing the z, y, and x
            dimensions. If a single int is provided, the same value will be used for z, y, and x.
        strides (tuple): Stride dimensions as a tuple of ints representing the z, y, and x
            strides. If a single int is provided, the same value will be used for z, y, and x.
        padding (str, optional): The type of padding to use. Can be "same" or "valid".
            if "same", the array is zero padded so that all the possible patches can be extracted.
            if "valid", the array is not padded and only patches that fully fit in the array are
            extracted. Defaults to "valid".
        device (str, optional): The device to use for prediction.
            Defaults to "cpu".

    Returns:
        arr_pred (np.ndarray): The predicted array.
    """

    model = model.eval()

    arr, pad_values = pad_3d_array(arr, patch_size, strides, padding, return_pad=True)

    if unpad and padding == "valid":
        Warning(
            "no padding was used, unpadding will have no effect, setting unpad to False"
        )
        unpad = False

    coords_top_corner, coords_bottom_corner = get_patch_coords(
        arr,
        patch_size,
        strides,
    )

    n_patches = coords_top_corner.shape[0]

    arr_pred = np.zeros(arr.shape)
    mask = np.zeros(arr.shape)

    for i in range(n_patches):
        top_corner = coords_top_corner[i]
        bottom_corner = coords_bottom_corner[i]

        patch = arr[
            top_corner[0] : bottom_corner[0],
            top_corner[1] : bottom_corner[1],
            top_corner[2] : bottom_corner[2],
        ]

        patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

        patch_pred = model(patch)

        patch_pred = patch_pred.squeeze(0).squeeze(0).cpu().detach().numpy()

        arr_pred[
            top_corner[0] : bottom_corner[0],
            top_corner[1] : bottom_corner[1],
            top_corner[2] : bottom_corner[2],
        ] += patch_pred

        mask[
            top_corner[0] : bottom_corner[0],
            top_corner[1] : bottom_corner[1],
            top_corner[2] : bottom_corner[2],
        ] += 1

    arr_pred = arr_pred / mask

    if unpad:
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = pad_values

        arr = arr[
            pad_front:-pad_back,
            pad_top:-pad_bottom,
            pad_left:-pad_right,
        ]
        return arr_pred

    return arr_pred


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
