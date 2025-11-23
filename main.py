import matplotlib.pyplot as plt
import metrax.nnx
import optax
import tensorflow as tf
from flax import nnx
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np

from dataset import convert_to_dataset, load_data, print_dataset_stats, get_folds
from metrax_monkey_patch import patch_metrax
from network import MLP

patch_metrax()
tf.random.set_seed(0)


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
    return preds


def preprocess_batch(batch):
    return {k: v for k, v in batch.items() if k != "path"}


def print_metrics(metrics: dict[str, float]):
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")


def train_and_evaluate(
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset | None = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    seed: int = 0,
):
    model = MLP(rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        accuracy=metrax.nnx.Accuracy(),
        loss=metrax.nnx.Average(),
        f1=metrax.nnx.FBetaScore(),
    )
    rngs = nnx.Rngs(seed)

    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_f1": [],
    }

    last_test_preds = []
    last_test_labels = []
    last_test_paths = []

    for epoch in range(epochs):
        for step, batch in enumerate(train_ds.as_numpy_iterator()):
            model.train()
            batch = preprocess_batch(batch)
            train_step(model, optimizer, metrics, rngs, batch)

        for metric, value in metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value)
        metrics.reset()

        if test_ds is not None:
            model.eval()
            for test_batch in test_ds.as_numpy_iterator():
                test_batch_paths = test_batch["path"]
                test_batch = preprocess_batch(test_batch)
                preds = eval_step(model, metrics, rngs, test_batch)

                if epoch == epochs - 1:
                    last_test_preds.extend(preds)
                    last_test_labels.extend(test_batch["label"])
                    last_test_paths.extend(test_batch_paths)

            for metric, value in metrics.compute().items():
                metrics_history[f"test_{metric}"].append(value)
            metrics.reset()

        if (epoch + 1) % 10 == 0:
            if test_ds is not None:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Test F1: {metrics_history['test_f1'][-1]:.2f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train F1: {metrics_history['train_f1'][-1]:.2f}"
                )

    return model, metrics_history, last_test_preds, last_test_labels, last_test_paths


data_dir = "/data/plant-rl/offline"
df = load_data(f"{data_dir}/labeled_dataset.parquet")

# Cross Validation
n_splits = 5
cv_results = []
cv_confusion_data = []
cv_all_preds = []
cv_all_labels = []
cv_all_paths = []
num_epochs = 100
batch_size = 32

print(f"Starting {n_splits}-Fold Cross Validation...")

for i, train_df, val_df in get_folds(df, n_splits=n_splits):
    print(f"\nFold {i + 1}/{n_splits}")
    train_ds = convert_to_dataset(
        train_df, batch_size, shuffle=True, drop_remainder=True
    )
    val_ds = convert_to_dataset(val_df, batch_size, shuffle=False, drop_remainder=False)

    print_dataset_stats(train_ds, "train")
    print_dataset_stats(val_ds, "val")

    model, history, fold_preds, fold_labels, fold_paths = train_and_evaluate(
        train_ds, val_ds, epochs=num_epochs
    )
    cv_results.append(history)

    # Collect predictions for confusion matrix and visualization
    cv_confusion_data.append((fold_labels, fold_preds))
    cv_all_preds.extend(fold_preds)
    cv_all_labels.extend(fold_labels)
    cv_all_paths.extend(fold_paths)

# Plot CV Confusion Matrices
fig, axes = plt.subplots(1, n_splits, figsize=(20, 4))
for i, (labels, preds) in enumerate(cv_confusion_data):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=axes[i], colorbar=False)
    axes[i].set_title(f"Fold {i + 1}")

plt.tight_layout()
plt.savefig("cv_confusion_matrices.png")
print("CV confusion matrices saved to cv_confusion_matrices.png")


def plot_metrics(histories: list[dict[str, list[float]]], filename: str):
    """Plot metrics for multiple histories (e.g. CV folds) or a single history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["loss", "accuracy", "f1"]
    titles = ["Loss", "Accuracy", "F1"]

    for ax, metric, title in zip(axes, metrics, titles):
        # Despline
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(title)

        for dataset, color in [("train", "tab:blue"), ("test", "tab:orange")]:
            # Check if this dataset exists in the first history (assuming all are same)
            if f"{dataset}_{metric}" not in histories[0]:
                continue

            # Collect all runs for this metric/dataset
            all_runs = [h[f"{dataset}_{metric}"] for h in histories]

            # Plot individual runs if there are multiple (CV case)
            if len(histories) > 1:
                for run in all_runs:
                    ax.plot(run, color=color, alpha=0.2)

            # Plot mean
            mean_run = np.mean(all_runs, axis=0)
            ax.plot(
                mean_run,
                color=color,
                label=f"Mean {dataset}" if len(histories) > 1 else dataset,
            )

        ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Metrics saved to {filename}")


# Aggregate and Plot CV Results
plot_metrics(cv_results, "cv_metrics.png")


# Visualize CV Predictions
cv_all_preds = np.array(cv_all_preds)
cv_all_labels = np.array(cv_all_labels)
cv_all_paths = np.array(cv_all_paths)

indices = np.random.permutation(len(cv_all_preds))[:25]
test_preds = cv_all_preds[indices]
test_labels = cv_all_labels[indices]
test_paths = cv_all_paths[indices]

fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, (ax, pred, label, path) in enumerate(
    zip(axs.flatten(), test_preds, test_labels, test_paths)
):
    path = path.decode("utf-8")
    full_path = f"{data_dir}/{path}"
    img = Image.open(full_path)
    ax.imshow(img)
    ax.set_title(f"label={label}, pred={pred}")
    ax.axis("off")
plt.savefig("cv_predictions.png")
print("CV predictions saved to cv_predictions.png")


# Final Model Training
print("\nTraining Final Model on Full Dataset...")
full_ds = convert_to_dataset(df, batch_size=32, shuffle=True, drop_remainder=True)
final_model, final_history, final_preds, final_labels, final_paths = train_and_evaluate(
    full_ds, None, epochs=num_epochs
)

# Plot Final Model Metrics
plot_metrics([final_history], "final_model_metrics.png")


# TODO: Save final model
