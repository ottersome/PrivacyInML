import os
import random
import sys
from itertools import chain
from pathlib import Path

new_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(new_path))

from pathlib import Path

import numpy as np
import torch
from munch import Munch
from PIL import Image
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

parent = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent))
from pml.utils import create_logger  # type: ignore

"""
StarGan Start
"""


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.logger = create_logger(__class__.__name__)
        self.preprocess()

        if mode == "train":
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, "r")]
        all_attr_names = lines[1].split()

        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        self.logger.info(f"Working with CelebA with {len(self.attr2idx)} attributes")
        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        # CHECK: some attributes are mutually exlusive. e.g. blonde/black hair.
        # this might affect our performance so could be a pain point to resolve later.

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == "1")

            if (i + 1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(
    image_dir,
    attr_path,
    selected_attrs,
    crop_size,
    image_size=128,
    batch_size=16,
    dataset="CelebA",
    mode="train",
    num_workers=1,
):
    """Build and return a data loader."""
    transform = []
    if mode == "train":
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == "CelebA":
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == "RaFD":
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
    )
    return data_loader


"""
My Old stuff
"""


# class CelebADataLoader(DataLoader):
#     """
#     DataLoader
#     For CelebA Dataset
#     """
#
#     def __init__(
#         self, dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True
#     ):
#         super(CelebADataLoader, self).__init__(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             drop_last=drop_last,
#         )
#
#
# class CelebADataset(Dataset):
#     """
#     Dataset
#     For CelebA DataseConvVAE, UNett
#     """
#
#     NAMES = ["train_ds.pt", "val_ds.pt", "test_ds.pt"]
#
#     def __init__(self, dataset: List, root: str, transform):
#         assert len(dataset) > 0, "Empty dataset?"
#         self.root = root
#         self.transform = transform
#         self.dataset = dataset
#         pass  # TODO:
#
#     def __getitem__(self, index):
#         filename, label = self.dataset[index]
#         image = Image.open(os.path.join(self.root, "imgs/", filename))
#         image_bw = image.convert("L")  # TODO: add option if found necessary
#         return self.transform(image_bw), torch.FloatTensor(label)
#
#     def set_mode(self, mode: Mode):
#         self.mode = mode
#
#     def __len__(self):
#         return len(self.dataset)
#
#
# class DataModule(L.LightningDataModule):
#     NAMES = ["train_ds.pt", "val_ds.pt", "test_ds.pt"]
#
#     def __init__(
#         self,
#         root: str,
#         attr_path: str,
#         cache_path: str,
#         selected_attrs: List[str],
#         transform,
#         batch_size: int,
#         split_percents: List[float],
#     ):
#         super().__init__()
#         self.root = root
#         self.attr_path = attr_path
#         self.split_percents = split_percents
#         self.cache_path = cache_path
#         self.batch_size = batch_size
#
#         self.logger = setup_logger(
#             __class__.__name__, log_path=Path(__file__).resolve().parent
#         )
#
#         self.selected_attrs = selected_attrs
#         self.transform = transform
#         self.mode = Mode.TRAIN
#
#         self.train_dataset = None
#         self.val_dataset = None
#         self.test_dataset = None
#
#         self.attr2idx = {}
#         self.idx2attr = {}
#
#         self.image_width = -1
#         self.image_height = -1
#         self.channels = -1
#
#     def setup(self, stage: str):
#         self.pre_prepare_data()
#
#     def prepare_data(self):
#         check_files = [exists(join(self.cache_path, name)) for name in self.NAMES]
#         if all(check_files):  # already cached
#             train_list, val_list, test_list = self._load_cached_lists()
#             self.logger.info("Cached Dataset found. Loading...")
#         else:
#             self.logger.info("Found no cached dataset, will preprocess...")
#             train_list, val_list, test_list = self._preprocess_lists()
#
#         self.train_dataset = CelebADataset(train_list, self.root, self.transform)
#         self.val_dataset = CelebADataset(val_list, self.root, self.transform)
#         self.test_dataset = CelebADataset(test_list, self.root, self.transform)
#
#     def pre_prepare_data(self):
#         # Load Stuff
#         needs_to_load_data = [
#             self.train_dataset == None,
#             self.val_dataset == None,
#         ]
#         if any(needs_to_load_data):
#             self.pre_prepare_data()
#
#     def _load_cached_lists(self):
#         self.logger.info("Loading cached data")
#         # TODO: ensure this loads the new objects
#         train_list = torch.load(join(self.cache_path, self.NAMES[0]))
#         val_list = torch.load(join(self.cache_path, self.NAMES[1]))
#         test_list = torch.load(join(self.cache_path, self.NAMES[2]))
#
#         sample_image = Image.open(os.path.join(self.root, "imgs", train_list[0][0]))
#         self.image_width = sample_image.size[0]
#         self.image_height = sample_image.size[1]
#         self.channels = len(sample_image.getbands())
#         assert self.channels == 3 or self.channels == 1
#
#         self.logger.info(
#             f"The amount of samples are:\n\t- {len(train_list)} for Training \n\t- {len(val_list)} for Validation \n\t- {len(test_list)} for Testing"
#         )
#         return train_list, val_list, test_list
#
#     def _preprocess_lists(self):
#         """
#         Preprocess CelebA Dataset
#         """
#         self.logger.info("Preprocess the CelebA Dataset")
#         assert os.path.exists(self.root), "Cannot find root path for data"
#         assert os.path.exists(self.attr_path), "Cannot find property file"
#
#         train_list, val_list, test_list = ([], [], [])
#
#         # Read attribut
#         lines = [line.rstrip() for line in open(self.attr_path, "r")]
#
#         # Check Existance of Field
#         sample_image = Image.open(os.path.join(self.root, "imgs", lines[2].split()[0]))
#
#         self.image_height = sample_image.size[0]
#         self.image_width = sample_image.size[1]
#         self.channels = len(sample_image.getbands())
#
#         all_attr_names = lines[1].split()
#         for i, attr_name in enumerate(all_attr_names):
#             self.attr2idx[attr_name] = i
#             self.idx2attr[i] = attr_name
#         self.logger.info(f"Samples available {len(lines) - 2}")
#
#         lines = lines[2:]
#         random.seed(1234)
#         random.shuffle(lines)
#         nts = len(lines)  # num total samps
#
#         self.logger.info("Adding samples")
#         bar = tqdm(lines, desc="Adding Samples")
#         for i, line in enumerate(lines):
#             split = line.split()
#             filename = split[0]
#             values = split[1:]
#             label = []
#
#             file_path = os.path.join(self.root, "imgs", filename)
#             if not os.path.exists(file_path):
#                 self.logger.warn(f"Ignoring image {file_path} because not found.")
#                 bar.update(1)
#                 continue
#             # Check size
#             img = Image.open(file_path)
#             assert (
#                 img.size[0] == self.image_height and img.size[1] == self.image_width
#             ), "Not matching sizes"
#
#             for attr_name in self.selected_attrs:
#                 idx = self.attr2idx[attr_name]
#                 label.append(values[idx] == "1")
#             if (i + 1) <= nts * self.split_percents[0]:
#                 train_list.append([filename, label])
#             if (i + 1) > nts * self.split_percents[0] and (i + 1) < nts * sum(
#                 self.split_percents[:2]
#             ):
#                 val_list.append([filename, label])
#             else:
#                 test_list.append([filename, label])
#             bar.update(1)
#
#         self.logger.info(
#             f"All samples constructed, saving the datasets to cache dir {self.cache_path}"
#         )
#         # Save all datasets for later use
#         os.makedirs(self.cache_path, exist_ok=True)
#         torch.save(train_list, join(self.cache_path, self.NAMES[0]))
#         torch.save(val_list, join(self.cache_path, self.NAMES[1]))
#         torch.save(test_list, join(self.cache_path, self.NAMES[2]))
#
#         self.logger.info("Finished preprocessing the CelebA dataset...")
#         return train_list, val_list, test_list
#
#     # Overwrites
#     def train_dataloader(self):
#         return CelebADataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             num_workers=12,
#             shuffle=True,  # CHECK: we might not want this to compare images for sake of repeatability
#             drop_last=True,  # , num_workers=1
#         )
#
#     # Overwrites
#     def val_dataloader(self):
#         return CelebADataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=12,
#             shuffle=True,
#             drop_last=True,  # , num_workers=1
#         )
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=12,
#             collate_fn=collate_fn,
#             shuffle=True,  # Avoids ugly validation loss graph
#         )
#
#         return Munch({k: v.to(self.device) for k, v in inputs.items()})
