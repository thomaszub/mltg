import jax.numpy as jnp
from epsilon import ConstantEpsilon, DecayingEpsilon
from flax import nnx


def test_constant_epsilon() -> None:
    rngs = nnx.Rngs(epsilon=42)
    epsilon = ConstantEpsilon(rngs, 0.4)
    samples = epsilon.sample((5000,))
    assert jnp.mean(samples) == 0.40039998


def test_decaying_epsilon() -> None:
    rngs = nnx.Rngs(epsilon=42)
    epsilon = DecayingEpsilon(rngs, 1.0, 0.0, 0.5)
    samples = epsilon.sample((5000,))
    assert jnp.mean(samples) == 1

    epsilon.update()
    samples = epsilon.sample((5000,))
    assert jnp.mean(samples) == 0.50479996

    epsilon.update()
    samples = epsilon.sample((5000,))
    assert jnp.mean(samples) == 0.25079998
