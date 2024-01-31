import logging
import os

import matplotlib.pyplot as plt
from deepface import DeepFace
from torch import Tensor


def setup_logger(logger_name: str, level=logging.INFO, log_path=__file__):
    os.makedirs(os.path.join(log_path, "logs/"), exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh = logging.FileHandler(os.path.join(log_path, "logs/", logger_name + ".log"), "w")
    sh = logging.StreamHandler()
    fh.setLevel(logging.DEBUG)
    sh.setLevel(level)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def create_logger(logger_name: str, log_path=None, level=logging.INFO):
    """
    Bit more convenient version of the above. Dont want to modify above cuz it breaks a lot
    """
    if log_path == None:
        log_path = os.getcwd()

    os.makedirs(os.path.join(log_path, "logs/"), exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh = logging.FileHandler(os.path.join(log_path, "logs/", logger_name + ".log"), "w")
    sh = logging.StreamHandler()
    fh.setLevel(logging.DEBUG)
    sh.setLevel(level)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def show_image(image: Tensor):
    """
    Take CxHxW image and show it
    """

    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    plt.close()
