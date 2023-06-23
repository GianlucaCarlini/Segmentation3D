import os
import numpy as np
import shutil


def train_test_split(
    data_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    save_dir: str = None,
):
    """Split a dataset into train and test sets.

    Args:
        data_dir (str): The path to the data directory.
        test_size (float, optional): The proportion of the dataset to include in the test split.
            Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before applying
            the split. Pass an int for reproducible output across multiple function calls.
            Defaults to 42.
        shuffle (bool, optional): Whether or not to shuffle the data before splitting.
            Defaults to True.
        save_dir (str, optional): The path to the directory where the split data will be saved.
            Defaults to None.

    Returns:
        train_ids (list): The list of training ids.
        test_ids (list): The list of testing ids.
    """

    ids = os.listdir(data_dir)

    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(ids)

    split_idx = int(len(ids) * (1 - test_size))

    train_ids = ids[:split_idx]
    test_ids = ids[split_idx:]

    if save_dir is None:
        save_dir = "./"

    os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "test"), exist_ok=True)

    train_dir = os.path.join(save_dir, "train")
    test_dir = os.path.join(save_dir, "test")

    for train_id in train_ids:
        print(f"Copying {train_id} to {train_dir}... \r", end="", flush=True)

        shutil.copy(os.path.join(data_dir, train_id), os.path.join(train_dir, train_id))

    for test_id in test_ids:
        print(f"Copying {test_id} to {test_dir}... \r", end="", flush=True)

        shutil.copy(os.path.join(data_dir, test_id), os.path.join(test_dir, test_id))


if __name__ == "__main__":
    train_test_split(
        data_dir="./data",
        test_size=0.2,
        random_state=42,
        shuffle=True,
        save_dir="./",
        verbose=True,
    )
