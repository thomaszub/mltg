from collections.abc import Callable

import grain
import jax
import jax.numpy as jnp


class TextDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, max_sequence_length: int, texts: list[str], encoder: Callable[[str], list[int]]) -> None:
        self.max_sequence_length = max_sequence_length
        self.texts = texts
        self.encoder = encoder

    def __getitem__(self, idx: int) -> jax.Array:
        text = self.texts[idx]
        tokens = self.encoder(text)
        if len(tokens) > self.max_sequence_length:
            tokens = tokens[: self.max_sequence_length]
        tokens.extend([0] * (self.max_sequence_length - len(tokens)))
        return jnp.array(tokens)

    def __len__(self) -> int:
        return len(self.texts)
