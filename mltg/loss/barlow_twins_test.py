import jax.numpy as jnp
import pytest
from jax import grad

from mltg.loss.barlow_twins import (
    WrongFirstAxisSizeError,
    WrongShapeError,
    barlow_twins_loss,
    calc_cross_correlation,
)


def test_barlow_twins_loss() -> None:
    z1 = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    z2 = jnp.array([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])

    loss = barlow_twins_loss(z1, z2)

    assert loss.shape == ()
    assert jnp.isclose(loss, 5.56, atol=1e-5), f"loss {loss} not 0"


def test_barlow_twins_loss_zero_lambda() -> None:
    z1 = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    z2 = jnp.array([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])

    loss = barlow_twins_loss(z1, z2, lambda_param=0.0)

    assert jnp.isfinite(loss).all()
    assert loss.shape == ()
    assert jnp.isclose(loss, 50.0 / 9.0, atol=1e-5)


def test_barlow_twins_loss_gradient_flow() -> None:
    z1 = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    z2 = jnp.array([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])

    grad_loss = grad(barlow_twins_loss, argnums=(0, 1))
    grads_z1, grads_z2 = grad_loss(z1, z2)

    assert grads_z1.shape == z1.shape
    assert grads_z2.shape == z2.shape
    assert jnp.all(jnp.isfinite(grads_z1))
    assert jnp.all(jnp.isfinite(grads_z2))


def test_calc_cross_correlating() -> None:
    z1 = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    z2 = jnp.array([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])

    cc = calc_cross_correlation(z1, z2)

    assert cc.shape == (2, 2)
    assert jnp.isclose(
        cc,
        jnp.array([[-2.0 / 3.0, -2.0 / 3.0], [-2.0 / 3.0, -2.0 / 3.0]]), atol=1e-5
    ).all()


# Error test cases

def test_barlow_twins_loss_shape_mismatch() -> None:
    z1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    z2 = jnp.array([[1.0, 2.0]])  # Different shape

    with pytest.raises(WrongShapeError):
        barlow_twins_loss(z1, z2)


def test_barlow_twins_loss_batch_size_mismatch() -> None:
    z1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    z2 = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    with pytest.raises(WrongShapeError):
        barlow_twins_loss(z1, z2)


def test_calc_cross_correlating_shape_mismatch() -> None:
    z1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    z2 = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    with pytest.raises(WrongFirstAxisSizeError):
        calc_cross_correlation(z1, z2)
