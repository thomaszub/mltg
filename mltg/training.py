from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx

type MetricFn = Callable[[nnx.Module, jax.Array, jax.Array], jax.Array]

# JIT compile the training and evaluation steps
def get_train_step(
        optimizer: nnx.Optimizer,
        loss_fn: MetricFn,
) -> Callable[
    [nnx.Module, jax.Array, jax.Array],
    tuple[nnx.Param, optax.OptState, float],
]:
    def train_step(
            model: nnx.Module,
            x: jax.Array,
            y: jax.Array
    ) -> tuple[nnx.Param, optax.OptState, float]:
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        optimizer.update(model, grads)
        return loss
    return train_step

def get_eval_step(
        loss_fn: MetricFn,
        accuracy_fn: MetricFn,
) -> Callable[
    [nnx.Module, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
]:
    def eval_step(
            model: nnx.Module,
            x: jax.Array,
            y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        loss = loss_fn(model, x, y)
        acc = accuracy_fn(model, x, y)
        return loss, acc
    return eval_step


def train(# noqa: PLR0913
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        loss_fn: MetricFn,
        accuracy_fn: MetricFn,
        num_epochs: int,
        batch_size: int,
        train_data: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        random_seed: int = 42
) -> tuple[list[float], list[float], list[float], list[float]]:
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    eval_step = get_eval_step(loss_fn, accuracy_fn)
    train_step = get_train_step(optimizer, loss_fn)
    train_images, train_labels, test_images, test_labels = train_data

    for epoch in range(num_epochs):
        # Shuffle training data
        rngs = nnx.Rngs(random_seed + epoch)
        indices = jnp.arange(len(train_images))
        shuffled_indices = jax.random.permutation(rngs(), indices)

        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_batches = 0

        for i in range(0, len(train_images), batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]
            x_batch = train_images[batch_indices]
            y_batch = train_labels[batch_indices]

            batch_loss = train_step(model, x_batch, y_batch)
            batch_acc = accuracy_fn(model, x_batch, y_batch)

            epoch_train_loss += batch_loss
            epoch_train_acc += batch_acc
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches
        avg_train_acc = epoch_train_acc / num_batches

        # Evaluate on test set
        test_loss, test_acc = eval_step(model, test_images, test_labels)

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: train_loss={avg_train_loss:.4f}, "
            f"train_acc={avg_train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, "
            f"test_acc={test_acc:.4f}"
        )

    return train_losses, train_accuracies, test_losses, test_accuracies
