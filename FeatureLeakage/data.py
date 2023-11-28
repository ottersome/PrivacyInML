"""
Creates Dataset
"""
import os
from pathlib import Path, PosixPath

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

order = ["age", "gender", "race", "datentime"]


def preprocess_image(image):
    # Normalize Image,
    image = image / 255.0
    # Subtract by mean
    image = image - image.mean()
    return image


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
