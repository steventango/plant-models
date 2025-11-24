import argparse
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import metrax.nnx
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
from flax import nnx
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

from dataset import convert_to_dataset, get_folds, load_data, print_dataset_stats
from metrax_monkey_patch import patch_metrax
from network import MLP
from plot import plot_metrics, plot_predictions

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
        logging.info(f"{metric}: {value:.2f}")


def train_and_evaluate(
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset | None = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    seed: int = 0,
    description: str = "Training",
):
    model = MLP(rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=metrax.nnx.Average(),
        f1=metrax.nnx.FBetaScore(),
    )
    rngs = nnx.Rngs(seed)

    metrics_history = {
        "train_loss": [],
        "train_f1": [],
        "test_loss": [],
        "test_f1": [],
    }

    last_test_preds = []
    last_test_labels = []
    last_test_paths = []

    pbar = tqdm(range(epochs), desc=description, unit="epoch")
    for epoch in pbar:
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

        # Update progress bar
        postfix = {}
        if test_ds is not None:
            postfix["test_f1"] = f"{metrics_history['test_f1'][-1]:.2f}"
        else:
            postfix["train_f1"] = f"{metrics_history['train_f1'][-1]:.2f}"
            postfix["train_loss"] = f"{metrics_history['train_loss'][-1]:.2f}"

        pbar.set_postfix(postfix)

    return model, metrics_history, last_test_preds, last_test_labels, last_test_paths


def configure_logging(output_dir: Path):
    """Configures logging to file and console."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(output_dir / "summary.log")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)


def save_model(final_model, output_dir):
    ckpt = {"model": nnx.state(final_model)}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save((output_dir / "model").resolve(), ckpt, force=True)
    logging.info(f"Final model saved to {output_dir / 'model'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/mlp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite existing output directory"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.force:
            raise ValueError(
                f"Output directory {output_dir} already exists. Use --force to overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir)

    data_dir = Path("/data/plant-rl/offline")
    df = load_data(str(data_dir / "labeled_dataset.parquet"))

    # Cross Validation
    n_splits = args.n_splits
    num_epochs = args.epochs
    batch_size = 32
    cv_results = []
    cv_confusion_data = []
    cv_all_preds = []
    cv_all_labels = []
    cv_all_paths = []

    logging.info(f"Starting {n_splits}-Fold Cross Validation...")

    for i, train_df, val_df in get_folds(df, n_splits=n_splits):
        logging.info(f"\nFold {i + 1}/{n_splits}")
        train_ds = convert_to_dataset(
            train_df, batch_size, shuffle=True, drop_remainder=True
        )
        val_ds = convert_to_dataset(
            val_df, batch_size, shuffle=False, drop_remainder=False
        )

        print_dataset_stats(train_ds, "train")
        print_dataset_stats(val_ds, "val")

        model, history, fold_preds, fold_labels, fold_paths = train_and_evaluate(
            train_ds, val_ds, epochs=num_epochs, description=f"Fold {i + 1}"
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
    plt.savefig(output_dir / "cv_confusion_matrices.png")
    logging.info(
        f"CV confusion matrices saved to {output_dir / 'cv_confusion_matrices.png'}"
    )

    # Print CV Summary Statistics
    logging.info("\n=== Cross Validation Summary ===")
    metrics_to_report = ["test_f1", "test_loss"]
    for metric in metrics_to_report:
        # Get the final value for each fold
        values = [h[metric][-1] for h in cv_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        logging.info(f"Mean {metric}: {mean_val:.4f} ± {std_val:.4f}")

    # Plot Aggregated Confusion Matrix
    cm_agg = confusion_matrix(cv_all_labels, cv_all_preds)
    disp_agg = ConfusionMatrixDisplay(confusion_matrix=cm_agg, display_labels=[0, 1])
    fig_agg, ax_agg = plt.subplots(figsize=(6, 6))
    disp_agg.plot(ax=ax_agg, colorbar=False)
    ax_agg.set_title("Aggregated Confusion Matrix (All Folds)")
    plt.tight_layout()
    plt.savefig(output_dir / "cv_confusion_matrix_aggregated.png")
    logging.info(
        f"Aggregated confusion matrix saved to {output_dir / 'cv_confusion_matrix_aggregated.png'}"
    )

    plot_metrics(cv_results, "cv_metrics.png", output_dir)
    plot_predictions(cv_all_preds, cv_all_labels, cv_all_paths, data_dir, output_dir)

    logging.info("\nTraining Final Model on Full Dataset...")
    full_ds = convert_to_dataset(df, batch_size=32, shuffle=True, drop_remainder=True)
    final_model, final_history, final_preds, final_labels, final_paths = (
        train_and_evaluate(full_ds, None, epochs=num_epochs, description="Final Model")
    )
    plot_metrics([final_history], "final_metrics.png", output_dir)
    save_model(final_model, output_dir)


if __name__ == "__main__":
    main()
