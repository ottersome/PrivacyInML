import logging
import os

import matplotlib.pyplot as plt
from torch import Tensor


def create_logger(name: str, level=logging.INFO):
    os.makedirs("./logs/", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh = logging.FileHandler(os.path.join("./logs", name + ".log"))
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
