import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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
    output_path = output_dir / filename
    plt.savefig(output_path)
    logging.info(f"Metrics saved to {output_path}")


def plot_predictions(
    preds, labels, paths, data_dir, output_dir, filename="cv_predictions.png", n=64
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
    output_path = output_dir / filename
    plt.savefig(output_path)
    logging.info(f"Predictions saved to {output_path}")
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
    output_path = output_dir / filename
    plt.savefig(output_path)
    logging.info(f"Aggregated confusion matrix saved to {output_path}")


def plot_trajectories(
    trajectories,
    data_dir,
    output_dir,
    policy_name="policy",
    filename="trajectories.pdf",
):
    """
    Plot K trajectories as a grid.

    Each row is a trajectory, each column is a timestep.

    Args:
        trajectories: List of trajectories, where each trajectory is a list of dicts
                     containing 'image_path', 'action', 'area', 'similarity'
        data_dir: Base directory for images
        output_dir: Output directory for plot
        policy_name: Name of the policy used
        filename: Output filename
    """
    if not trajectories:
        logging.warning("No trajectories to plot")
        return

    n_trajectories = len(trajectories)
    n_steps = max(len(t) for t in trajectories)

    # Create figure: n_trajectories rows, n_steps columns (max 7 per row)
    MAX_COLS = 7
    cols = min(n_steps, MAX_COLS)
    rows_per_traj = int(np.ceil(n_steps / cols))
    total_rows = n_trajectories * rows_per_traj

    fig, axs = plt.subplots(
        total_rows, cols, figsize=(cols * 2, total_rows * 2), squeeze=False
    )

    # Turn off axes for all subplots initially
    for ax in axs.flat:
        ax.axis("off")

    for traj_idx, trajectory in enumerate(trajectories):
        for step_idx, step in enumerate(trajectory):
            # Calculate row and col
            row_in_traj = step_idx // cols
            col_in_traj = step_idx % cols

            abs_row = traj_idx * rows_per_traj + row_in_traj
            abs_col = col_in_traj

            ax = axs[abs_row, abs_col]

            # Load and display image
            image_path = step["image_path"]
            if isinstance(image_path, (bytes, np.bytes_)):
                image_path = image_path.decode("utf-8")
            full_path = f"{data_dir}/{image_path}"

            try:
                img = Image.open(full_path)
                ax.imshow(img)

                title = f"t={step_idx}\n"
                area_val = step.get("dataset_area")
                if area_val is not None:
                    title += f"Area: {area_val:.0f}"
                else:
                    title += "Area: N/A"

                ax.set_title(title, fontsize=8)
            except Exception as e:
                logging.warning(f"Could not load image {full_path}: {e}")
                ax.text(
                    0.5,
                    0.5,
                    "Image\nNot Found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches="tight")
    logging.info(f"Rollout trajectories plot saved to {output_path}")
    plt.close(fig)
