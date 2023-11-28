import os
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from typing import List

import pandas as pd
import torch
from jax import grad
from jax import numpy as jnp
from jax import random
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import UTKDataset, utk_parse
from model import VGGRegressionModel


def apfun():
    ap = ArgumentParser()
    # ap.add_argument("--prototypes", default=10, type=int, help="Amount of Prototypes")
    ap.add_argument(
        "--data_path", default="./data/utkface", type=str, help="Path to dataset"
    )
    ap.add_argument(
        "--batch_size", default=32, type=int, help="Batch Size for Training"
    )
    ap.add_argument("--epochs", default=100, type=int, help="Amount of Epochs")
    ap.add_argument("--train_proportion", default=0.8)

    # TODO: maybe add dataset source
    return ap.parse_args()


if __name__ == "__main__":
    args = apfun()

    # Load The Data
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Path {data_path} does not exist")
    imgs, ages, genders, races = utk_parse(data_path)

    # Partition Data
    ts = int(args.train_proportion * len(imgs))
    imgs_train, imgs_test = imgs[:ts], imgs[ts:]
    labels_train, labels_test = (ages[:ts], ages[ts:])

    # Dataset and DataLoader
    ds_train = UTKDataset(imgs_train, labels_train)
    ds_test = UTKDataset(imgs_test, labels_test)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size)
    num_batches = ceil(len(ds_train) / args.batch_size)

    # Train the model
    model = VGGRegressionModel()
    optim = torch.optim.Adam(model.parameters())

    # Train loop
    ebar = tqdm(range(args.epochs))
    dl_iter = iter(dl_train)
    criterium = torch.nn.MSELoss()
    for batch in iter(dl_train):
        optim.zero_grad()
        batch_x, batch_y = next(dl_iter)

        pred = model(batch_x)

        loss = criterium(pred, batch_y)
        loss.backward()
        optim.step()

        ebar.set_description(f"Loss: {loss.item():.2f}")
        ebar.update()
