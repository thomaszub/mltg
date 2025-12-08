from collections.abc import Sequence
from typing import Protocol

from flax import nnx


class Epsilon(Protocol):
    def sample(self, shape: Sequence[int] = (1,)) -> bool:
        pass

    def update(self) -> None:
        pass


class ConstantEpsilon(Epsilon):
    def __init__(self, rngs: nnx.Rngs, epsilon: float) -> None:
        super().__init__()
        self._rngs = rngs
        self._epsilon = epsilon

    def sample(self, shape: Sequence[int] = (1,)) -> bool:
        return self._rngs.epsilon.uniform(shape) < self._epsilon

    def update(self) -> None:
        pass


class DecayingEpsilon(Epsilon):
    def __init__(
        self,
        rngs: nnx.Rngs,
        epsilon_start: float,
        epsilon_end: float,
        decay_rate: float,
    ) -> None:
        super().__init__()
        self._rngs = rngs
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon = epsilon_start
        self._decay_rate = decay_rate

    def sample(self, shape: Sequence[int] = (1,)) -> bool:
        return self._rngs.epsilon.uniform(shape) < self._epsilon

    def update(self) -> None:
        self._epsilon = max(self._decay_rate * self._epsilon, self._epsilon_end)
