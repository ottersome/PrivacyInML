# Below is a simple linear-softmax implementation in JAX:
import logging
import math
import os
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, nn
from tqdm import tqdm

from model_funcs import loss_fn, sgd, softmax
from utils import parse_faces, setup_logger

plt.style.use("rose-pine")


# %% Definition of essentials
def apfun():
    ap = ArgumentParser()
    ap.add_argument("--ds_path", default="./faces")
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--weights_dir", default="./weights", type=str)
    ap.add_argument("--batch_size", default=16, type=int)

    return ap.parse_args()


# %% Declare global structures
args = apfun()
os.makedirs(args.weights_dir, exist_ok=True)
logger = setup_logger("main", logging.DEBUG)
key = jax.random.PRNGKey(42)
key, WKey, BKey = jax.random.split(key, 3)

initializer = nn.initializers.glorot_normal()
W = initializer(WKey, (10304, 40), dtype=jnp.float32)
b = initializer(BKey, (1, 40), jnp.float32)

# Import the AT&T faces dataset
logger.info("Loading images into ds")
train_ds_n_labels, test_ds_n_labels = parse_faces(args.ds_path)

# Shuffle the train ds
logger.info("Formatting into JAX friendly.")
logger.info(f" Using {len(train_ds_n_labels)} samples")

np.random.shuffle(train_ds_n_labels)
# Make it into a form usable by JAX
ds = jnp.array([img for img, _ in train_ds_n_labels]).squeeze()
labels = jnp.array([lbl for _, lbl in train_ds_n_labels])


# %% Softmax Model and Components
logger.info("Creating SoftMax Model")


# %% Training loop
epoch_bar = tqdm(total=args.epochs, desc="Epoch")
total_batches = math.ceil(ds.shape[0] / args.batch_size)

test_ds = jnp.array([img for img, _ in test_ds_n_labels]).squeeze()
test_labels = jnp.array([lbl for _, lbl in test_ds_n_labels])
cur_accuracy = 0
for e in range(args.epochs):
    loss = 0
    for x in range(total_batches):
        i = x * args.batch_size
        j = np.min([i + args.batch_size, len(ds)])

        batched_ds = ds[i:j, :]
        batched_labels = labels[i:j]
        loss = loss_fn(W, b, batched_ds, batched_labels)

        # SGD
        W, b = sgd(W, b, batched_ds, batched_labels)

    # Validation every epoch
    preds = softmax(jnp.dot(test_ds, W) + b).squeeze()
    cur_accuracy = jnp.sum(jnp.argmax(preds, axis=1) == test_labels) / preds.shape[0]
    epoch_bar.set_description(f"Epoch: {e} Loss: {loss}, Accuracy {cur_accuracy}")
    # Test the model
    epoch_bar.update(1)

# Final Specific Accuracy Test
accs = []
preds = softmax(jnp.dot(test_ds, W) + b).squeeze()
for i in range(40):
    idxs = test_labels == i
    cur_accuracy = jnp.mean(jnp.argmax(preds[idxs, :], axis=1) == test_labels[idxs])
    accs.append(cur_accuracy)
# Plot histogram of accs
logger.info(f"Accuracies across the board are {accs}")

plt.bar(np.arange(len(accs)), accs)
plt.show()


logger.info(f"final accuracy for 39 is {cur_accuracy}")
# %% Save the weigths
logger.info("Saving weights")

jnp.save(os.path.join(args.weights_dir, "W.npy"), W)
jnp.save(os.path.join(args.weights_dir, "b.npy"), b)

logger.info("Done")
