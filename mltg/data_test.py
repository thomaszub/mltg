import jax.numpy as jnp
from data import dataloader, sample
from flax import nnx


def test_dataloader() -> None:
    rngs = nnx.Rngs(42)
    batch_size = 2
    x = jnp.array([[1, 2], [3, 4], [5, 6]])
    y = jnp.array([7, 8, 9])
    loader = dataloader((x, y), batch_size, rngs)
    xs, ys = loader.__next__()
    assert (xs == jnp.array([[3, 4], [1, 2]])).all()
    assert (ys == jnp.array([8, 7])).all()
    xs, ys = loader.__next__()
    assert (xs == jnp.array([[5, 6]])).all()
    assert (ys == jnp.array([9])).all()
    loader.close()


def test_sample() -> None:
    rngs = nnx.Rngs(42)
    batch_size = 2
    x = jnp.array([[1, 2], [3, 4], [5, 6]])
    y = jnp.array([7, 8, 9])
    xs, ys = sample((x, y), batch_size, rngs)
    assert (xs == jnp.array([[3, 4], [1, 2]])).all()
    assert (ys == jnp.array([8, 7])).all()
