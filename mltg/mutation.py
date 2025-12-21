from collections.abc import Callable

import jax
from flax import nnx
from jax.tree_util import GetAttrKey


class Mutator(Callable[[tuple[type[GetAttrKey], ...], nnx.Param], nnx.Param]):
    def __call__(
        self, path: tuple[type[GetAttrKey], ...], param: nnx.Param
    ) -> nnx.Param:
        pass


class GaussianMutator(Mutator):
    def __init__(self, rngs: nnx.Rngs, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.rngs = rngs
        self.mu = mu
        self.sigma = sigma

    def __call__(
        self, _: tuple[type[GetAttrKey], ...], param: nnx.Param
    ) -> nnx.Param:
        return param + nnx.Param(
            self.mu + self.sigma * self.rngs.mutation.normal(param.shape)
        )


def apply_mutation(model: nnx.Pytree, mutator: Mutator) -> nnx.Pytree:
    return jax.tree.map_with_path(mutator, model)
