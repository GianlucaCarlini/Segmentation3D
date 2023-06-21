import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
import os


class PatchDataloader(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        patch_size: tuple = None,
        threshold: float = None,
        transform=None,
        preprocessing=None,
    ):
        """Dataset for loading patches from images and labels. It can be used
        for lazy loading of patches from images and labels. It samples a random
        index from the image and label and extracts a patch of the specified
        size. If the number of non-zero voxels in the label is less than the
        threshold, it will sample another index until the threshold is met.

        Args:
            images_dir (str): path to the image directory
            labels_dir (str): path to the label directory
            patch_size (tuple, optional): Size of the patches to load as a tuple of ints
                representing the x, y, and z dimension. If a single int is provided,
                the same value will be used for x, y, and z Defaults to None.
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

        self.ids = os.listdir(self.images_dir)

        self.images = [os.path.join(self.images_dir, image_id) for image_id in self.ids]
        self.labels = [os.path.join(self.labels_dir, image_id) for image_id in self.ids]

        self.transform = transform
        self.preprocessing = preprocessing

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

        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(self.labels[index])
        file_reader.ReadImageInformation()

        x, y, z = file_reader.GetSize()

        while True:
            extract_idx_x = np.random.randint(0, max(x - self.patch_size[0], 1))
            extract_idx_y = np.random.randint(0, max(y - self.patch_size[1], 1))
            extract_idx_z = np.random.randint(0, max(z - self.patch_size[2], 1))

            file_reader.SetExtractIndex((extract_idx_x, extract_idx_y, extract_idx_z))
            file_reader.SetExtractSize(
                (self.patch_size[0], self.patch_size[1], self.patch_size[2])
            )

            label = file_reader.Execute()
            label = sitk.GetArrayFromImage(label)

            if np.sum(label > 0.0) > self.threshold * (
                self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
            ):
                break

        file_reader.SetFileName(self.images[index])

        image = file_reader.Execute()
        image = sitk.GetArrayFromImage(image)

        if self.preprocessing:
            image = self.preprocessing(image)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return self.len
