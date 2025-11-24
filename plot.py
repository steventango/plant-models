import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image
from sklearn.metrics._classification import confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay


def plot_metrics(
    histories: list[dict[str, list[float]]], filename: str, output_dir: Path
):
    """Plot metrics for multiple histories (e.g. CV folds) or a single history."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    metrics = ["loss", "f1"]
    titles = ["Loss", "F1"]

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
    plt.savefig(output_dir / filename)
    logging.info(f"Metrics saved to {output_dir / filename}")


def plot_predictions(
    preds, labels, paths, data_dir, output_dir, filename="cv_predictions.png", n=25
):
    """Visualizes a random sample of predictions."""
    preds = np.array(preds)
    labels = np.array(labels)
    paths = np.array(paths)

    if len(preds) > n:
        indices = np.random.permutation(len(preds))[:n]
        preds = preds[indices]
        labels = labels[indices]
        paths = paths[indices]

    grid_size = int(np.ceil(np.sqrt(len(preds))))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    for i, ax in enumerate(axs.flatten()):
        if i < len(preds):
            pred, label, path = preds[i], labels[i], paths[i]
            path = path.decode("utf-8")
            full_path = f"{data_dir}/{path}"
            img = Image.open(full_path)
            ax.imshow(img)
            ax.set_title(f"label={label}, pred={pred}")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / filename)
    logging.info(f"Predictions saved to {output_dir / filename}")
    plt.close(fig)


def plot_confusion_matrix(
    labels: list[int],
    preds: list[int],
    output_dir: Path,
    filename: str = "cv_confusion_matrix_aggregated.png",
):
    cm_agg = confusion_matrix(labels, preds)
    disp_agg = ConfusionMatrixDisplay(confusion_matrix=cm_agg, display_labels=[0, 1])
    fig_agg, ax_agg = plt.subplots(figsize=(6, 6))
    disp_agg.plot(ax=ax_agg, colorbar=False)
    ax_agg.set_title("Aggregated Confusion Matrix (All Folds)")
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    logging.info(f"Aggregated confusion matrix saved to {output_dir / filename}")
