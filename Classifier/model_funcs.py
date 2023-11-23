from jax import grad
from jax import numpy as jnp


# Define the softmax model
def old_softmax(
    x,
):  # TOREM: This one should not be working. But alas it give better accuracy
    """Compute softmax values for each sets of scores in x."""
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / e_x.sum(axis=0)


def softmax(x):  # TEST: It should be this way
    """Compute softmax values for each sets of scores in x."""
    # e_x = jnp.exp(x)
    e_x = jnp.exp(x - jnp.max(x))  # CHECK: shold i use this one?
    sum = jnp.expand_dims(e_x.sum(axis=1), 1)
    return e_x / sum


# Define the loss function
def loss_fn(W: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray):
    dot = jnp.dot(x, W)
    non_scalar = -jnp.log(
        softmax(dot + b)[jnp.arange(dot.shape[0]), y]
    )  # This is wrong
    return jnp.mean(non_scalar)  # + 0.1 * jnp.sum(W**2)


# Define the gradient function
grad_loss_W = grad(loss_fn, argnums=0)
grad_loss_b = grad(loss_fn, argnums=1)


# Define the optimizer
def sgd(W, b, x, y, lr=1e-2):
    # Backpop
    newW = W - lr * grad_loss_W(W, b, x, y)
    newB = b - lr * grad_loss_b(W, b, x, y)
    return newW, newB
