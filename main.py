from dataset import convert_to_dataset
from dataset import split_data
from dataset import print_dataset_stats
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
from flax import nnx
from IPython.display import clear_output

from dataset import load_data
from network import CNN

tf.random.set_seed(0)


data_dir = "/data/plant-rl/offline"
df = load_data(f"{data_dir}/labeled_dataset.parquet")

train_groups = [(13, 1)]
train_df, test_df = split_data(df, train_groups)

train_steps = 1200
batch_size = 32
train_ds = convert_to_dataset(train_df, "train", batch_size, train_steps)
test_ds = convert_to_dataset(test_df, "test", batch_size)
print_dataset_stats(train_ds, "train")
print_dataset_stats(test_ds, "test")


model = CNN(rngs=nnx.Rngs(0))
nnx.display(model)


learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum), wrt=nnx.Param)
nnx.display(optimizer)

metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model: CNN, rngs: nnx.Rngs, batch):
    logits = model(batch["image"], rngs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: CNN,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, rngs, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch):
    loss, logits = loss_fn(model, rngs, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}

rngs = nnx.Rngs(0)

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    model.train()
    train_step(model, optimizer, metrics, rngs, batch)

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
        for metric, value in metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value)
        metrics.reset()

        model.eval()
        for test_batch in test_ds.as_numpy_iterator():
            eval_step(model, metrics, rngs, test_batch)

        for metric, value in metrics.compute().items():
            metrics_history[f"test_{metric}"].append(value)
        metrics.reset()

        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title("Loss")
        ax2.set_title("Accuracy")
        for dataset in ("train", "test"):
            ax1.plot(metrics_history[f"{dataset}_loss"], label=f"{dataset}_loss")
            ax2.plot(
                metrics_history[f"{dataset}_accuracy"], label=f"{dataset}_accuracy"
            )
        ax1.legend()
        ax2.legend()
        plt.show()


model.eval()


@nnx.jit
def pred_step(model: CNN, batch):
    logits = model(batch["image"])
    return logits.argmax(axis=1)


test_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(model, test_batch)

fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(test_batch["image"][i, ..., 0], cmap="gray")
    ax.set_title(f"label={pred[i]}")
    ax.axis("off")
