from collections.abc import Generator

import jax
import jax.numpy as jnp
from flax import nnx


def dataloader(
    arrays: tuple[jax.Array, jax.Array], batch_size: int, rngs: nnx.Rngs
) -> Generator[tuple[jax.Array, jax.Array]]:
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    perm = rngs.data.permutation(indices)
    start = 0
    end = min(batch_size, dataset_size)
    while start < dataset_size:
        batch_perm = perm[start:end]
        yield arrays[0][batch_perm], arrays[1][batch_perm]
        start = end
        end = min(dataset_size, end + batch_size)


def sample(
    arrays: tuple[jax.Array, jax.Array], batch_size: int, rngs: nnx.Rngs
) -> tuple[jax.Array, jax.Array]:
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    perm = rngs.data.permutation(indices)
    end = min(batch_size, dataset_size)
    batch_perm = perm[0:end]
    return arrays[0][batch_perm], arrays[1][batch_perm]
