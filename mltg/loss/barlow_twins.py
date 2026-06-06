import jax
import jax.numpy as jnp
from jax import Array


class WrongFirstAxisSizeError(ValueError):
    def __init__(self, size1: int, size2: int) -> None:
        msg = f"First axis of inputs (size={size1} and (size={size2}) must be equal"
        super().__init__(msg)


class WrongShapeError(ValueError):
    def __init__(self, shape1: tuple[int, ...], shape2:tuple[int, ...]) -> None:
        msg = f"Input shapes (shape={shape1}) and (shape={shape2}) must be equal"
        super().__init__(msg)

@jax.jit
def calc_cross_correlation(z1: jax.Array, z2: jax.Array) -> jax.Array:
    """
    Computes the cross-correlation matrix between two datasets along the first axis.

    Normalizes both input matrices by subtracting their respective means and dividing
    by the standard deviation of the first input (z1) along axis 0.

    *Note*: The unbiased standard deviation estimator 1/N-1.

    :param z1: First input matrix of shape (n_samples, n_features1).
    :param z2: Second input matrix of shape (n_samples, n_features2).
    :return: Cross-correlation matrix of shape (n_features1, n_features2) where each
             element (i,j) represents the correlation between feature i of z1 and
             feature j of z2.
    :raises WrongFirstAxisSizeError: If the first axis of z1 and z2 have different sizes.
    """
    if z1.shape[0] != z2.shape[0]:
        raise WrongFirstAxisSizeError(z1.shape[0], z2.shape[0])
    size = z1.shape[0]
    z1norm = (z1 - jnp.mean(z1, axis=0)) / jnp.std(z1, axis=0, ddof=1)
    z2norm = (z2 - jnp.mean(z2, axis=0)) / jnp.std(z1, axis=0, ddof=1)
    return jnp.einsum("bi,bj->ij", z1norm, z2norm) / size

@jax.jit(static_argnames="lambda_param")
def barlow_twins_loss(z1: jax.Array, z2: jax.Array, lambda_param: float=0.005) -> Array:
    """
    Computes the Barlow Twins loss for self-supervised learning.

    This loss function encourages the cross-correlation matrix between two sets of
    embeddings (e.g., from augmented views of the same input) to be close to the
    identity matrix. It consists of two terms:
    1. **Invariant term**: Maximizes the diagonal elements of the cross-correlation
       matrix (encouraging feature invariance across views).
    2. **Redundancy reduction term**: Minimizes the off-diagonal elements of the
       cross-correlation matrix (reducing feature redundancy).

    The total loss is a weighted sum of these terms, where the weight for the
    redundancy term is controlled by `lambda_param`.

    :param z1: First set of embeddings (e.g., from one augmented view).
               Shape: (batch_size, embedding_dim).
    :param z2: Second set of embeddings (e.g., from another augmented view).
               Shape: (batch_size, embedding_dim).
    :param lambda_param: Weight for the redundancy reduction term.
                         Controls the trade-off between invariance and redundancy.
                         Default: 0.005 (as suggested in the original paper).
    :return: Scalar Barlow Twins loss value.
    :raises WrongShapeError: If z1 and z2 have different shapes.
    """

    if z1.shape != z2.shape:
        raise WrongShapeError(z1.shape, z2.shape)

    # Cross-correlation matrix
    c = calc_cross_correlation(z1, z2)

    # Invariant term: sum of diagonal elements (maximize)
    invariant_term = jnp.sum((1.0 - jnp.diag(c))**2)

    # Redundancy reduction term: sum of off-diagonal elements squared (minimize)
    off_diag = jnp.fill_diagonal(c, jnp.array([[0]]), inplace=False)
    redundancy_term = jnp.sum(off_diag**2)

    # Total loss
    return invariant_term + lambda_param * redundancy_term
