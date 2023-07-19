import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
import os
from typing import Union, Callable, Any
from torchio.data import Subject
import torchio as tio

__all__ = ["PatchDataloader"]


def gaussian_sampling(volume_size, patch_size, std_factor=8):
    if patch_size[0] < 0:
        return 0, 0, 0

    x, y, z = volume_size

    x_offset = patch_size[0] / 2
    y_offset = patch_size[1] / 2
    z_offset = patch_size[2] / 2

    center_x = x / 2 - x_offset
    center_y = y / 2 - y_offset
    center_z = z / 2 - z_offset

    std_x = x / std_factor
    std_y = y / std_factor
    std_z = z / std_factor

    extract_idx_x = np.random.normal(loc=center_x, scale=std_x)
    extract_idx_y = np.random.normal(loc=center_y, scale=std_y)
    extract_idx_z = np.random.normal(loc=center_z, scale=std_z)

    extract_idx_x = int(np.clip(extract_idx_x, 0, x - patch_size[0]))
    extract_idx_y = int(np.clip(extract_idx_y, 0, y - patch_size[1]))
    extract_idx_z = int(np.clip(extract_idx_z, 0, z - patch_size[2]))

    return extract_idx_x, extract_idx_y, extract_idx_z


def uniform_sampling(volume_size, patch_size):
    if patch_size[0] < 0:
        return 0, 0, 0

    x, y, z = volume_size

    if patch_size is None:
        patch_size = (0, 0, 0)

    extract_idx_x = np.random.randint(0, max(x - patch_size[0], 1))
    extract_idx_y = np.random.randint(0, max(y - patch_size[1], 1))
    extract_idx_z = np.random.randint(0, max(z - patch_size[2], 1))

    return extract_idx_x, extract_idx_y, extract_idx_z


sampling_functions = {"uniform": uniform_sampling, "gaussian": gaussian_sampling}


class PatchDataloader(Dataset):
    """
    Dataset for loading patches from images and labels. It can be used
    for lazy loading of patches from images and labels. It samples a random
    index from the image and label and extracts a patch of the specified
    size. If the number of non-zero voxels in the label is less than the
    threshold, it will sample another index until the threshold is met

    Parameters
    ----------
    images_dir : str
        path to the image directory
    labels_dir : str
        path to the label directory
    patch_size : tuple | int, optional
        Size of the patches to load as a tuple of ints
        representing the x, y, and z dimension. If a single int is provided,
        the same value will be used for x, y, and z.
        If less than 0, the whole volume is used. Defaults to None.
    sampling_method : str, optional
        Sampling method to use. Can be either
        "uniform" or "gaussian". Defaults to "uniform".
    threshold : float, optional
        Threshold value to consider for patch sampling.
        If the sum of non-zero pixels in the sampled patch is lower than
        threshold, then another patch is sampled until the threshold condition is met
        Defaults to None.
    transform : callable, optional
        Optional transform to apply to the image and label.
        The same transform is applied to both. Defaults to None.
    preprocessing : callable, optional
        Optional preprocessing to apply to the image.
        Defaults to None.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        patch_size: Union[tuple, int] = None,
        sampling_method: str = "uniform",
        threshold: float = None,
        transform: Callable = None,
        preprocessing: Callable = None,
        positional: bool = False,
        repeat: int = 1,
        **kwargs,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.ids = os.listdir(self.labels_dir)

        if repeat > 1:
            self.ids = self.ids * repeat

        self.images = [os.path.join(self.images_dir, image_id) for image_id in self.ids]
        self.labels = [os.path.join(self.labels_dir, image_id) for image_id in self.ids]

        if preprocessing is not None:
            self.preprocessing = preprocessing
        else:
            self.preprocessing = None
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

        self.positional = positional

        self.reader = sitk.ImageFileReader()

        self.sampling_method = sampling_method

        self.sampling_function = sampling_functions.get(self.sampling_method, None)

        if self.sampling_function is None:
            raise NotImplementedError(
                f"Sampling method {self.sampling_method} not implemented, available methods are: {sampling_functions.keys()}"
            )

        if patch_size is not None:
            if isinstance(patch_size, int):
                self.patch_size = (patch_size, patch_size, patch_size)
            else:
                self.patch_size = patch_size
        else:
            self.patch_size = (96, 96, 96)

        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = 0.0

        self.len = len(self.ids)

        self.kwargs = kwargs

    def __getitem__(self, index):
        self.reader.SetFileName(self.labels[index])
        self.reader.ReadImageInformation()

        self.x, self.y, self.z = self.reader.GetSize()

        if self.patch_size[0] < 0:
            patch_size = (self.x, self.y, self.z)
        else:
            patch_size = self.patch_size

        while True:
            (
                self.extract_idx_x,
                self.extract_idx_y,
                self.extract_idx_z,
            ) = self.sampling_function(
                volume_size=(self.x, self.y, self.z),
                patch_size=patch_size,
                **self.kwargs,
            )

            self.reader.SetExtractIndex(
                (self.extract_idx_x, self.extract_idx_y, self.extract_idx_z)
            )
            self.reader.SetExtractSize((patch_size[0], patch_size[1], patch_size[2]))

            label = self.reader.Execute()
            label = sitk.GetArrayFromImage(label)

            if np.sum(label > 0.0) > self.threshold * (
                patch_size[0] * patch_size[1] * patch_size[2]
            ):
                break

        self.reader.SetFileName(self.images[index])

        image = self.reader.Execute()
        image = sitk.GetArrayFromImage(image)

        image = np.expand_dims(image, axis=0)

        if self.preprocessing is not None:
            image = self.preprocessing(image)

        if self.transform is not None:
            image, label = self.transform(image, label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        if self.positional:
            rel_idx = self.get_relative_index()
            rel_idx = torch.from_numpy(np.array(rel_idx)).float()

            return image, label, rel_idx

        return image, label

    def __len__(self):
        return self.len

    def get_relative_index(self):
        rel_x = self.extract_idx_x / self.x
        rel_y = self.extract_idx_y / self.y
        rel_z = self.extract_idx_z / self.z

        return [[rel_z], [rel_y], [rel_x]]


class PatchLoaderIO(PatchDataloader):
    """
    Dataset for loading patches from images and labels. It can be used
    for lazy loading of patches from images and labels. It samples a random
    index from the image and label and extracts a patch of the specified
    size. If the number of non-zero voxels in the label is less than the
    threshold, it will sample another index until the threshold is met.
    This version of the dataset can be used to apply torchio transforms
    to both image and label.

    A note on why I don't use torchio's Queue. TorchIO loads the whole volumes,
    performs the transformations and then extracts the patches. This is extremely
    slow, at least in my experience. This dataset instead loads the patches directly,
    and performs the transformations on the patches. This is much faster.
    The only drawback is that statistics as mean and variance cannot be computed on
    the whole volume, but only on the patches, so it is necessary to compute them in
    advance and pass them to the Loader.
    In my experience, this approach is an order of magnitude faster than using torchio's
    Queue. However, it could also be that I was not able to make Queue work properly.

    Parameters
    ----------
    images_dir : str
        path to the image directory
    labels_dir : str
        path to the label directory
    patch_size : tuple | int, optional
        Size of the patches to load as a tuple of ints
        representing the x, y, and z dimension. If a single int is provided,
        the same value will be used for x, y, and z.
        If less than 0, the whole volume is used. Defaults to None.
    sampling_method : str, optional
        Sampling method to use. Can be either
        "uniform" or "gaussian". Defaults to "uniform".
    threshold : float, optional
        Threshold value to consider for patch sampling.
        If the sum of non-zero pixels in the sampled patch is lower than
        threshold, then another patch is sampled until the threshold condition is met
        Defaults to None.
    transform : callable, optional
        Optional transform to apply to the image and label.
        The same transform is applied to both. Defaults to None.
    preprocessing : callable, optional
        Optional preprocessing to apply to the image.
        Defaults to None.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        patch_size: Union[tuple, int] = None,
        sampling_method: str = "uniform",
        threshold: float = None,
        transform: Callable[..., Any] = None,
        preprocessing: Callable[..., Any] = None,
        positional: bool = False,
        repeat: int = 1,
        **kwargs,
    ):
        super().__init__(
            images_dir,
            labels_dir,
            patch_size,
            sampling_method,
            threshold,
            transform,
            preprocessing,
            positional,
            repeat,
            **kwargs,
        )

    def __getitem__(self, index):

        self.reader.SetFileName(self.labels[index])
        self.reader.ReadImageInformation()

        self.x, self.y, self.z = self.reader.GetSize()

        if self.patch_size[0] < 0:
            patch_size = (self.x, self.y, self.z)
        else:
            patch_size = self.patch_size

        while True:
            (
                self.extract_idx_x,
                self.extract_idx_y,
                self.extract_idx_z,
            ) = self.sampling_function(
                volume_size=(self.x, self.y, self.z),
                patch_size=patch_size,
                **self.kwargs,
            )

            self.reader.SetExtractIndex(
                (self.extract_idx_x, self.extract_idx_y, self.extract_idx_z)
            )
            self.reader.SetExtractSize((patch_size[0], patch_size[1], patch_size[2]))

            label = self.reader.Execute()
            label = sitk.GetArrayFromImage(label)

            if np.sum(label > 0.0) > self.threshold * (
                patch_size[0] * patch_size[1] * patch_size[2]
            ):
                break
        
        label = np.expand_dims(label, axis=0)

        self.reader.SetFileName(self.images[index])

        image = self.reader.Execute()
        image = sitk.GetArrayFromImage(image)

        image = np.expand_dims(image, axis=0)

        if self.preprocessing is not None:
            image = self.preprocessing(image)

        self.subject = Subject(image=tio.ScalarImage(tensor=image), label=tio.LabelMap(tensor=label))

        if self.transform is not None:
            self.subject = self.transform(self.subject)

        image = self.subject["image"][tio.DATA].float()
        label = self.subject["label"][tio.DATA].float()

        if self.positional:
            rel_idx = self.get_relative_index()
            rel_idx = torch.from_numpy(np.array(rel_idx)).float()

            return image, label, rel_idx

        return image, label

    def get_subject(self):
        return self.subject       
