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

# plt.style.use("rose-pine")


def apfun():
    ap = ArgumentParser()
    ap.add_argument("--ds_path", default="./faces")
    ap.add_argument("--epochs", default=10000, type=int)
    ap.add_argument(
        "--recon_dir", default="./recons", type=str, help="Dir to place reconstructions"
    )
    ap.add_argument("--subject", default=-1, type=int, help="Subject to run this into.")
    ap.add_argument(
        "--gif",
        action="store_true",
        help="Whether or not to create a gif out of this",
    )
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
        ]
        + penalty(x)
    )  # HACK: change 40 to softcoded


# Unecessary but better to keep images well formatted
def penalty(x):
    """
    In order to make sure pixel values are between 0 and 1
    """
    # Get element-wise max between x and an array of 1:
    return jnp.mean(
        jnp.square(jnp.maximum(jnp.zeros_like(x), x - 1))
        + jnp.square(jnp.minimum(jnp.zeros_like(x), x))
    )


grad_loss = grad(loss_fn, argnums=0)


def sgd(x, W, b, label, lr=0.001):
    gl = grad_loss(x, W, b, label)
    logger.info(f"Gradient min and max are {gl.min()} and {gl.max()}")
    grado = lr * gl
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

    # Exampl of Subject you want:
    subject_example = trueths[1, :].reshape(112, 92)

    # Reversal Loop
    logger.info(f"Working with label {label}")
    ebar = tqdm(total=args.epochs, desc="Reconstruction Epoch", leave=False)
    # Evaluating on real images

    # x = initializer(GKey, (1, 10304), dtype=jnp.float32)
    x = jnp.zeros((1, 10304))
    # Initialize X to be gaussian (mostly) withon 0 and 1 (element wise) with variance of 0.1
    # x = random.normal(GKey, (1, 10304), dtype=jnp.float32) * 0.1 + 0.5
    # Ititialize x to b uniform in 0-1
    # x = random.uniform(GKey, (1, 10304), dtype=jnp.float32)
    for e in range(args.epochs):
        # Get Predictions
        loss = loss_fn(x, W, b, label)

        # x = sgd(x, W, b, label)
        # Same as above but clipped
        # x = jnp.clip(sgd(x, W, b, label), 0, 1)
        x = sgd(x, W, b, label)
        if args.gif:
            img = x.reshape((112, 92))
            plt.plot()
            plt.title("Reconstruction")
            plt.imshow(img, cmap="gray")
            plt.savefig(os.path.join(args.recon_dir, f"{label}_{e}.png"))
            plt.close()

        ebar.set_description(f"Reconstruction Epoch Loss {loss}")
        ebar.update(1)

    # x = x.mean(axis=0).reshape((92, 112))
    x = x.reshape((112, 92))
    # Show x and subject_example as image side by side in pylot
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(subject_example, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(x, cmap="gray")
    # Dont show but save instead
    plt.savefig(os.path.join(args.recon_dir, f"{label}.png"))
    # plt.show()


if __name__ == "__main__":
    # Scoop out all labels
    dirlist = list(os.listdir(args.ds_path))
    dirbar = tqdm(total=len(dirlist), desc="Iterating through sample", leave=True)

    os.makedirs(args.recon_dir, exist_ok=True)

    # Load the weights
    W = jnp.load("./weights/W.npy")
    b = jnp.load("./weights/b.npy")
    if args.subject > 0:
        main_loop(args.subject, W, b)
    else:
        for dir in dirlist:
            id = int(re.sub("[a-zA-Z]", "", dir)) - 1
            dirbar.set_description(f"Iterating through sample {id}")
            main_loop(id, W, b)
