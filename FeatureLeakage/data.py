"""
Creates Dataset
"""
import random
from logging import DEBUG
from math import ceil
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import create_logger, show_image

order = ["age", "gender", "race", "datentime"]

data_logger = create_logger("data", DEBUG)


def preprocess_image(image):
    # Normalize Image,
    image = image / 255.0
    # Subtract by mean
    image = image - image.mean()
    return image


def utk_parse_federated(
    path: Path,
    batch_size,
    num_clients: int,
    batch_of_insertion=7,
    tt_split=0.8,
    race_of_interest=1,
):
    """
    Returns
    -------
        trains: List of samples for each client
        test: overall test dataset
        seggretated_data: seggretated races, to be used in extraneous logic
    """
    features, ages, _, races = utk_parse(path)
    # We start with shuffling and corresponding labels
    df = pd.DataFrame([features, ages, races], columns=["features", "ages", "race"])
    # Take out all "races" = 1 to a different df
    segg_df = df[df["races"] == race_of_interest]
    seggretated_batches = ceil(len(segg_df) / batch_size)
    data_logger.info(
        f"We have {len(segg_df)} seggretated examples which at batch size{batch_size}"
        f" will yield {seggretated_batches} seggretated batches starting from {batch_of_insertion}"
    )
    # drop rows with "race_of_interest" in them
    df = df.drop(segg_df.index)

    # Shuffle df and do train-test split
    df = df.sample(frac=1)
    df_train = df[: int(tt_split * len(df))]
    df_test = df[int(tt_split * len(df)) :]

    # Partition dataset for federated clients
    cds = ceil(len(df) / num_clients)  # Client data size
    clients_df_train = []
    for c in range(num_clients):
        clients_df_train.append(
            df_train.iloc[c * (cds) : (c + 1) * (cds)].reset_index(drop=True)
        )

    return clients_df_train, df_test, segg_df


def utk_parse(path: Path):
    """
    Args
    ----
        path: We exepect images in path to be in utk format
    """
    samples = []
    ages, genders, races = [], [], []
    files = list(path.iterdir())
    fbar = tqdm(total=len(files), desc="Constructing Dataset")
    for file in files:
        if file.is_file() and file.name.endswith(".jpg"):
            # name is [age]_[gender]_[race]_[date&time].jpg
            # Extract the 4 features:
            split = {
                order[i]: int(n.strip().replace(".jpg.chip.jpg", ""))
                for i, n in enumerate(file.name.split("_"))
                # FIX: remove this weird jpg.chip from the files themselves
            }
            ages.append(torch.tensor(split["age"], dtype=torch.float32))
            genders.append(torch.tensor(split["gender"], dtype=torch.int))
            races.append(split["race"])
            # load the image
            img = cv2.imread(str(file))
            img = preprocess_image(img)
            img = torch.Tensor(img)
            img = img.permute(2, 0, 1)
            samples.append(img)
        fbar.update(1)
    return samples, ages, genders, races


class UTKDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]
