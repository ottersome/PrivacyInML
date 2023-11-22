# Below is a simple linear-softmax implementation in JAX:
import logging
import math
import os
import re
from argparse import ArgumentParser
from typing import List

import cv2 as cv
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, nn
from tqdm import tqdm


# %% Definition of essentials
def apfun():
    ap = ArgumentParser()
    ap.add_argument("--ds_path", default="./faces")
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--batch_size", default=16, type=int)

    return ap.parse_args()


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


def import_images(path, samples_train: List, samples_test: List):
    idx0 = int(re.sub("[a-zA-Z]", "", dir)) - 1
    for i, file in enumerate(os.listdir(path)):
        assert file.endswith(".png")
        # Import file as a grayscale numpy array
        img = (cv.imread(os.path.join(path, file), cv.IMREAD_GRAYSCALE) / 255).reshape(
            1, 10304
        )
        if i < 7:
            samples_train.append([img, idx0])
        else:
            samples_test.append([img, idx0])


# %% Declare global arrays
args = apfun()
logger = setup_logger("main", logging.DEBUG)
key = jax.random.PRNGKey(42)
key, WKey, BKey = jax.random.split(key, 3)

# %% Initial Data structures
# Create an empty array of 40x10x(92*112)
# array = jnp.empty((40, 10, 10304))

# %% Dataset parsing
# Import the AT&T faces dataset
logger.info("Loading images into ds")
train_ds = []
test_ds = []
dir_count = 0
if os.path.exists(args.ds_path):
    # Iterate over it
    dirlist = os.listdir(args.ds_path)
    dirlist.sort(key=lambda f: int(re.sub("[a-zA-Z]", "", f)))
    logger.debug(f"Amount of classes {len(dirlist)}")
    for dir in dirlist:
        dir_count += 1
        rel_dir = os.path.join(args.ds_path, dir)
        import_images(rel_dir, train_ds, test_ds)

assert dir_count == 40, f"You only checked on {dir_count}"
# Shuffle the train ds CHECK: I don't think this is necessary though(cuz no batching)
logger.info("Formatting into JAX friendly.")
logger.info(f" Using {len(train_ds)} samples")
np.random.shuffle(train_ds)
# Make it into a form usable by JAX
ds = jnp.array([img for img, _ in train_ds]).squeeze()
labels = jnp.array([lbl for _, lbl in train_ds])


# Create the Weight and Biases for the input
logger.info("Creating SoftMax Model")

initializer = nn.initializers.glorot_normal()
# TODO: maybe initialize them with  normal distribution or some other one
W = initializer(WKey, (10304, 40), dtype=jnp.float32)
# W = jnp.zeros((10304, 40))
b = initializer(BKey, (1, 40), jnp.float32)


# %% Softmax Model
# Define the softmax model
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = jnp.exp(x)
    return e_x / e_x.sum(axis=0)


# %% Train the coefficients
logger.info("Training SoftMax Model")


# Define the loss function
def loss_fn(W: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray):  # TOREM:
    dot = jnp.dot(x, W)
    non_scalar = -jnp.log(
        softmax(dot + b)[jnp.arange(dot.shape[0]), y]
    )  # This is wrong
    return jnp.mean(non_scalar) + 0.1 * np.sum(W**2)


# Define the gradient function
grad_loss_W = grad(loss_fn, argnums=0)
grad_loss_b = grad(loss_fn, argnums=1)


# Define the optimizer
def sgd(W, b, x, y, lr=1e-2):
    # Backpop
    newW = W - lr * grad_loss_W(W, b, x, y)
    newB = b - lr * grad_loss_b(W, b, x, y)
    return newW, newB


# %% Training loop
epoch_bar = tqdm(total=args.epochs, desc="Epoch")
total_batches = math.ceil(ds.shape[0] / args.batch_size)

tds = jnp.array([img for img, _ in test_ds])
tlbls = jnp.array([lbl for _, lbl in test_ds])
cur_accuracy = 0
for e in range(args.epochs):
    loss = 0
    for x in range(total_batches):
        i = x * args.batch_size
        j = np.min([i + args.batch_size, len(ds)])
        b_ds = ds[i:j, :]
        b_lbls = labels[i:j]
        loss = loss_fn(W, b, b_ds, b_lbls)
        # SGD
        W, b = sgd(W, b, b_ds, b_lbls)
    preds = softmax(jnp.dot(tds, W) + b).squeeze()
    cur_accuracy = jnp.sum(jnp.argmax(preds, axis=1) == tlbls) / preds.shape[0]
    epoch_bar.set_description(f"Epoch: {e} Loss: {loss}, Accuracy {cur_accuracy}")
    # Test the model
    epoch_bar.update(1)

# %% Testing the model
logger.info("Testing SoftMax Model")
# Test the model
logger.info("Predicting")
# Predict
# Get the accuracy
# acc = jnp.mean(jnp.argmax(preds, axis=1) == tlbls)

# %% Save the weigths
logger.info("Saving weights")
np.save("W.npy", W)
np.save("b.npy", b)

logger.info("Done")
