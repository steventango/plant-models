import matplotlib.pyplot as plt
import metrax.nnx
import optax
import tensorflow as tf
from flax import nnx
from PIL import Image

from metrax_monkey_patch import patch_metrax
from dataset import convert_to_dataset, load_data, print_dataset_stats, split_data
from network import MLP

patch_metrax()
tf.random.set_seed(0)


data_dir = "/data/plant-rl/offline"
df = load_data(f"{data_dir}/labeled_dataset.parquet")

train_groups = [(13, 1)]
train_df, test_df = split_data(df, train_groups)

train_epochs = 100
batch_size = 32
train_ds = convert_to_dataset(train_df, batch_size)
test_ds = convert_to_dataset(test_df, batch_size)
print_dataset_stats(train_ds, "train")
print_dataset_stats(test_ds, "test")


model = MLP(rngs=nnx.Rngs(0))
nnx.display(model)


learning_rate = 1e-3
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)
nnx.display(optimizer)

metrics = nnx.MultiMetric(
    accuracy=metrax.nnx.Accuracy(),
    loss=metrax.nnx.Average(),
    f1=metrax.nnx.FBetaScore(),
)


def loss_fn(model: MLP, rngs: nnx.Rngs, batch):
    logits = model(batch["embedding"], rngs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    preds = logits.argmax(axis=1)
    return loss, preds


@nnx.jit
def train_step(
    model: MLP,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(model, rngs, batch)
    metrics.update(values=loss, predictions=preds, labels=batch["label"])
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: MLP, metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch):
    loss, preds = loss_fn(model, rngs, batch)
    metrics.update(values=loss, predictions=preds, labels=batch["label"])


def preprocess_batch(batch):
    return {k: v for k, v in batch.items() if k != "path"}


def print_metrics(metrics: dict[str, float]):
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")


metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "train_f1": [],
    "test_loss": [],
    "test_accuracy": [],
    "test_f1": [],
}

rngs = nnx.Rngs(0)
for epoch in range(train_epochs):
    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        model.train()
        batch = preprocess_batch(batch)
        train_step(model, optimizer, metrics, rngs, batch)

    for metric, value in metrics.compute().items():
        metrics_history[f"train_{metric}"].append(value)
    metrics.reset()

    model.eval()
    for test_batch in test_ds.as_numpy_iterator():
        test_batch = preprocess_batch(test_batch)
        eval_step(model, metrics, rngs, test_batch)

    for metric, value in metrics.compute().items():
        metrics_history[f"test_{metric}"].append(value)
    metrics.reset()
    print_metrics({k: v[-1] for k, v in metrics_history.items()})


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.set_title("Loss")
ax2.set_title("Accuracy")
ax3.set_title("F1")

for dataset in ("train", "test"):
    ax1.plot(metrics_history[f"{dataset}_loss"], label=f"{dataset}_loss")
    ax2.plot(metrics_history[f"{dataset}_accuracy"], label=f"{dataset}_accuracy")
    ax3.plot(metrics_history[f"{dataset}_f1"], label=f"{dataset}_f1")
ax1.legend()
ax2.legend()
ax3.legend()
plt.savefig("metrics.png")

model.eval()


@nnx.jit
def pred_step(model: MLP, batch):
    logits = model(batch["embedding"])
    return logits.argmax(axis=1)


test_batch = test_ds.as_numpy_iterator().next()
pred_batch = preprocess_batch(test_batch)
pred = pred_step(model, pred_batch)

fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    path = test_batch["path"][i].decode("utf-8")
    full_path = f"{data_dir}/{path}"
    img = Image.open(full_path)
    ax.imshow(img)
    ax.set_title(f"label={pred[i]}")
    ax.axis("off")
plt.savefig("predictions.png")
