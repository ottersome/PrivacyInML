import enum
import os
import random

# Add imports in upper directory relative to this one
import sys
from os.path import exists, join
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

parent_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_path))
from pml.utils import setup_logger  # type: ignore


class Mode(enum.Enum):
    TRAIN = "train"
    VALIDATE = "validate"


class CelebADataLoader(DataLoader):
    """
    DataLoader
    For CelebA Dataset
    """

    def __init__(
        self, dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True
    ):
        super(CelebADataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )


class CelebADataset(Dataset):
    """
    Dataset
    For CelebA Dataset
    """

    NAMES = ["train_ds.pt", "val_ds.pt", "test_ds.pt"]

    def __init__(
        self,
        root: str,
        attr_path: str,
        cache_path: str,
        selected_attrs: List[str],
        transform,
        mode: Mode,
        split_percents: List[float],
    ):
        self.root = root
        self.attr_path = attr_path
        self.split_percents = split_percents
        self.cache_path = cache_path

        self.logger = setup_logger(
            __class__.__name__, log_path=Path(__file__).resolve().parent
        )

        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = Mode.TRAIN

        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []

        self.attr2idx = {}
        self.idx2attr = {}

        self.image_dim = [-1, -1]
        check_files = [exists(join(self.cache_path, name)) for name in self.NAMES]

        if all(check_files):  # already cached
            self._load_cache()
            self.logger.info("Cached Dataset found. Loading...")
        else:
            self.logger.info("Found no cached dataset, will preprocess...")
            self._preprocess()

        if mode == Mode.TRAIN:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.val_dataset)

    def _load_cache(self):
        self.logger.info("Loading cached data")
        self.train_dataset = torch.load(join(self.cache_path, self.NAMES[0]))
        self.val_dataset = torch.load(join(self.cache_path, self.NAMES[1]))
        self.test_dataset = torch.load(join(self.cache_path, self.NAMES[2]))

    def _preprocess(self):
        """
        Preprocess CelebA Dataset
        """
        self.logger.info("Preprocess the CelebA Dataset")
        assert os.path.exists(self.root), "Cannot find root path for data"
        assert os.path.exists(self.attr_path), "Cannot find property file"

        # Read attribut
        lines = [line.rstrip() for line in open(self.attr_path, "r")]

        sample_image = Image.open(os.path.join(self.root, "imgs", lines[2].split()[0]))
        self.image_dim[0] = sample_image.size[0]
        self.image_dim[1] = sample_image.size[1]

        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        self.logger.info(f"Samples available {len(lines) - 2}")

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        nts = len(lines)  # num total samps

        self.logger.info("Adding samples")
        bar = tqdm(lines, desc="Adding Samples")
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            label = []

            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == "1")
            if (i + 1) <= nts * self.split_percents[0]:
                self.train_dataset.append([filename, label])
            if (i + 1) > nts * self.split_percents[0] and (i + 1) < nts * sum(
                self.split_percents[:2]
            ):
                self.val_dataset.append([filename, label])
            else:
                self.test_dataset.append([filename, label])
            bar.update(1)

        self.logger.info(
            f"All samples constructed, saving the datasets to cache dir {self.cache_path}"
        )
        # Save all datasets for later use
        os.makedirs(self.cache_path, exist_ok=True)
        torch.save(self.train_dataset, join(self.cache_path, self.NAMES[0]))
        torch.save(self.val_dataset, join(self.cache_path, self.NAMES[1]))
        torch.save(self.test_dataset, join(self.cache_path, self.NAMES[2]))

        self.logger.info("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == Mode.TRAIN else self.val_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.root, "imgs/", filename))
        return self.transform(image), torch.FloatTensor(label)

    def set_mode(self, mode: Mode):
        self.mode = mode

    def __len__(self):
        if self.mode == Mode.TRAIN:
            return len(self.train_dataset)
        else:
            return len(self.val_dataset)
