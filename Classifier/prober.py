"""
script will probe into a loaded linear regression model to obtain 
image reconstructions given a particular label
"""
import logging
import os
import re
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from jax import grad, random
from tqdm import tqdm

from model_funcs import softmax
from utils import import_images, setup_logger


def apfun():
    ap = ArgumentParser()
    ap.add_argument("--ds_path", default="./faces")
    ap.add_argument("--epochs", default=100, type=int)
    return ap.parse_args()


logger = setup_logger("main", logging.DEBUG)
args = apfun()
key = random.PRNGKey(42)
GKey = random.split(key, 1)
initializer = jax.nn.initializers.glorot_normal()


def loss_fn(x, W, b, label: int):
    preds = softmax(jnp.dot(x, W) + b)
    # logger.info(f"Preds average is {str(jnp.mean(preds[:,label]))}")
    return jnp.mean(
        1
        - preds[
            :,
            label
            # jnp.arange(preds.shape[0]), jnp.full((1, 16), label)
        ]  # HACK: hardcoded batch size
    )  # HACK: change 40 to softcoded


grad_loss = grad(loss_fn, argnums=0)


def sgd(x, W, b, label, lr=0.9):
    grado = lr * grad_loss(x, W, b, label)
    return x - grado


def main_loop(label, W: jnp.ndarray, b: jnp.ndarray):
    logger.info(f"Trying to invert subject s{label}")

    # Load the dataset
    image_dir = os.path.join(args.ds_path, "s" + str(label + 1))
    samples = []
    import_images(image_dir, samples, [], train_parttn=1)
    actual_images = [img for img, _ in samples]
    trueths = jnp.array(actual_images).squeeze()
    preds = softmax(jnp.dot(trueths, W) + b)
    avg_pred = jnp.mean(jnp.argmax(preds, axis=1) == label)
    logger.info(f"🚇 Average pred is {avg_pred}")

    # Reversal Loop
    logger.info(f"Working with label {label}")
    ebar = tqdm(total=args.epochs, desc="Reconstruction Epoch", leave=False)
    # Evaluating on real images

    # x = initializer(GKey, (16, 10304), dtype=jnp.float32)
    x = jnp.zeros((16, 10304))
    for _ in range(args.epochs):
        # Get Predictions
        loss = 0
        for b in range(16):
            loss = loss_fn(x, W, b, label)

            x = sgd(x, W, b, label)

        ebar.set_description(f"Reconstruction Epoch Loss {loss}")
        ebar.update(1)
    x = x.mean(axis=0).reshape((92, 112))
    # Show X as an image. Where x : jnp.ndarray
    plt.figure()
    plt.imshow(x, cmap="gray")
    plt.show()


def reversal_loop():
    pass


if __name__ == "__main__":
    # Scoop out all labels
    dirlist = list(os.listdir(args.ds_path))
    dirbar = tqdm(total=len(dirlist), desc="Iterating through sample", leave=True)

    # Load the weights
    W = jnp.load("./weights/W.npy")
    b = jnp.load("./weights/b.npy")

    for dir in dirlist[:1]:
        id = int(re.sub("[a-zA-Z]", "", dir)) - 1
        dirbar.set_description(f"Iterating through sample {id}")
        main_loop(id, W, b)
