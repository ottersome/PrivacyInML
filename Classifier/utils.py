import logging
import os
import re
from typing import List

import cv2 as cv


def setup_logger(name, level=logging.INFO):
    # Make dir
    os.makedirs("./log/", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    fh = logging.FileHandler("./log/" + name + ".log", mode="w")
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def import_images(path, samples_train: List, samples_test: List, train_parttn=0.6):
    # Get number inside of "path"
    idx0 = int(re.findall(r"\d+", path)[0]) - 1

    list_of_images = list(os.listdir(path))
    num_images = len(list_of_images)
    num_train_images = int(num_images * train_parttn)
    for i, file in enumerate(list_of_images):
        # Import file as a grayscale numpy array
        img = (cv.imread(os.path.join(path, file), cv.IMREAD_GRAYSCALE) / 255).reshape(
            1, 10304
        )
        if i <= num_train_images:
            samples_train.append([img, idx0])
        else:
            samples_test.append([img, idx0])


def parse_faces(path):
    train_ds_n_labels = []
    test_ds_n_labels = []
    if os.path.exists(path):
        # Iterate over it
        for dir in os.listdir(path):
            rel_dir = os.path.join(path, dir)
            import_images(rel_dir, train_ds_n_labels, test_ds_n_labels)
    return train_ds_n_labels, test_ds_n_labels
