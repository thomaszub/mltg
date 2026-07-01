import jax.numpy as jnp

from mltg.language.source import TextDataSource


def char_encoder(text: str) -> list[int]:
    return [ord(c) for c in text]


def test_text_data_source_getitem() -> None:
    texts = ["", "abc", "abcdefg"]
    max_sequence_length = 5

    source = TextDataSource(max_sequence_length, texts, char_encoder)
    result = jnp.array([source[idx] for idx in range(len(texts))])

    expected = jnp.array([
        [0, 0, 0, 0, 0],
        [97, 98, 99, 0, 0],
        [97, 98, 99, 100, 101]
    ])
    assert jnp.array_equal(result, expected)

