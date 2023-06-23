import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
import os
from typing import Union, Callable


class PatchDataloader(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        patch_size: Union[tuple, int] = None,
        threshold: float = None,
        transform: Callable = None,
        preprocessing: Callable = None,
    ):
        """Dataset for loading patches from images and labels. It can be used
        for lazy loading of patches from images and labels. It samples a random
        index from the image and label and extracts a patch of the specified
        size. If the number of non-zero voxels in the label is less than the
        threshold, it will sample another index until the threshold is met.

        Args:
            images_dir (str): path to the image directory
            labels_dir (str): path to the label directory
            patch_size (tuple | int, optional): Size of the patches to load as a tuple of ints
                representing the x, y, and z dimension. If a single int is provided,
                the same value will be used for x, y, and z.
                If less than 0, the whole volume is used. Defaults to None.
            threshold (float, optional): Threshold value to consider for patch sampling.
                If the sum of non-zero pixels in the sampled patch is lower than
                threshold, then another patch is sampled until the threshold condition is met
                Defaults to None.
            transform (callable, optional): Optional transform to apply to the image and label.
                The same transform is applied to both. Defaults to None.
            preprocessing (callable, optional): Optional preprocessing to apply to the image.
                Defaults to None.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.ids = os.listdir(self.labels_dir)

        self.images = [os.path.join(self.images_dir, image_id) for image_id in self.ids]
        self.labels = [os.path.join(self.labels_dir, image_id) for image_id in self.ids]

        self.transform = transform
        self.preprocessing = preprocessing
        self.reader = sitk.ImageFileReader()

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

    def __getitem__(self, index):

        self.reader.SetFileName(self.labels[index])
        self.reader.ReadImageInformation()

        x, y, z = self.reader.GetSize()

        if self.patch_size[0] < 0:
            patch_size = (x, y, z)
        else:
            patch_size = self.patch_size

        while True:
            extract_idx_x = np.random.randint(0, max(x - patch_size[0], 1))
            extract_idx_y = np.random.randint(0, max(y - patch_size[1], 1))
            extract_idx_z = np.random.randint(0, max(z - patch_size[2], 1))

            self.reader.SetExtractIndex((extract_idx_x, extract_idx_y, extract_idx_z))
            self.reader.SetExtractSize(
                (patch_size[0], patch_size[1], patch_size[2])
            )

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

        if self.preprocessing:
            image = self.preprocessing(image)

        if self.transform:
            image, label = self.transform(image, label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

    def __len__(self):
        return self.len
