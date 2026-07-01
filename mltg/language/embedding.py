import jax
import jax.numpy as jnp
from flax import nnx


class TokenEmbedding(nnx.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int, rngs: nnx.Rngs) -> None:
        self.embedding = nnx.Embed(num_embeddings=vocabulary_size, features=embedding_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.embedding(x)


class PositionEmbedding(nnx.Module):
    def __init__(self, max_sequence_length: int, embedding_size: int, rngs: nnx.Rngs) -> None:
        self.embedding = nnx.Embed(num_embeddings=max_sequence_length, features=embedding_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        positions = jnp.arange(x.shape[1]).reshape(1, -1)
        return self.embedding(positions)
