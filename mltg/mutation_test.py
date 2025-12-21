import jax
import jax.numpy as jnp
from flax import nnx
from mutation import GaussianMutator, apply_mutation


def test_gaussian_mutator() -> None:
    model = [
        {
            "first": nnx.Param(jnp.array([1.0, 2.0])),
            "second": {
                "third": nnx.Param(jnp.array([3.0, 4.0])),
                "fourth": nnx.Param(jnp.array([5.0, 6.0])),
            },
        }
    ]

    expected = [
        {
            "first": nnx.Param(jnp.array([1.0600401, 2.0409367])),
            "second": {
                "third": nnx.Param(jnp.array([3.0629563, 4.06355])),
                "fourth": nnx.Param(jnp.array([5.0255756, 6.029643])),
            },
        }
    ]

    mutator = GaussianMutator(rngs=nnx.Rngs(mutation=0), mu=0.05, sigma=0.01)
    # noinspection PyTypeChecker
    actual = apply_mutation(model, mutator)

    actual_flatten, _ = jax.tree.flatten(actual)
    expected_flatten, _ = jax.tree.flatten(expected)
    for act_leaf, exp_leaf in zip(actual_flatten, expected_flatten, strict=True):
        assert (act_leaf == exp_leaf).all()
