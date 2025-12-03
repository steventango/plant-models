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
    # Detect available metrics from the first history
    # Metrics are stored as train_X and test_X, extract unique metric names
    all_keys = set(histories[0].keys())
    metric_names = set()
    for key in all_keys:
        if key.startswith("train_") or key.startswith("test_"):
            metric_name = key.split("_", 1)[1]
            metric_names.add(metric_name)

    metric_names = sorted(metric_names)  # Sort for consistent ordering
    n_metrics = len(metric_names)

    if n_metrics == 0:
        logging.warning("No metrics to plot")
        return

    # Create subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]  # Make it iterable

    for ax, metric in zip(axes, metric_names):
        # Despline
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Capitalize title (e.g., "loss" -> "Loss", "area_mae" -> "Area Mae")
        title = metric.replace("_", " ").title()
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(title)

        for dataset, color in [("train", "tab:blue"), ("test", "tab:orange")]:
            # Check if this dataset exists in the first history (assuming all are same)
            metric_key = f"{dataset}_{metric}"
            if metric_key not in histories[0]:
                continue

            # Collect all runs for this metric/dataset
            all_runs = [h[metric_key] for h in histories]

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
    plt.close(fig)


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
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))

    for i, ax in enumerate(axs.flatten()):
        if i < len(preds):
            pred, label, path = preds[i], labels[i], paths[i]
            path = path.decode("utf-8")
            full_path = f"{data_dir}/{path}"
            img = Image.open(full_path)
            ax.imshow(img)
            ax.set_title(f"label={label}\npred={pred}")
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


def plot_regression_predictions(
    preds: list[float],
    labels: list[float],
    output_dir: Path,
    filename: str = "cv_regression_predictions.png",
    title: str = "Predicted vs True Area",
):
    """Plots predicted vs true values for regression."""
    preds = np.array(preds)
    labels = np.array(labels)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(labels, preds, alpha=0.5)

    # Identity line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)

    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)

    plt.tight_layout()
    output_path = output_dir / filename
    plt.savefig(output_path)
    logging.info(f"Regression predictions plot saved to {output_path}")
    plt.close(fig)


def plot_regression_results(
    paths,
    true_areas,
    pred_areas,
    data_dir,
    output_dir,
    filename="regression_results.png",
    n=100,
):
    """Plots a grid of images with true and predicted areas."""
    paths = np.array(paths)
    true_areas = np.array(true_areas)
    pred_areas = np.array(pred_areas)

    if len(paths) > n:
        indices = np.random.permutation(len(paths))[:n]
        paths = paths[indices]
        true_areas = true_areas[indices]
        pred_areas = pred_areas[indices]

    grid_size = int(np.ceil(np.sqrt(len(paths))))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))

    for i, ax in enumerate(axs.flatten()):
        if i < len(paths):
            path = paths[i]
            if isinstance(path, (bytes, np.bytes_)):
                path = path.decode("utf-8")
            full_path = f"{data_dir}/{path}"
            try:
                img = Image.open(full_path)
                ax.imshow(img)
                ax.set_title(
                    f"True: {true_areas[i]:.0f}\nPred: {pred_areas[i]:.0f}",
                    fontsize=8,
                )
            except Exception as e:
                logging.warning(f"Could not load image {full_path}: {e}")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / filename
    plt.savefig(output_path)
    logging.info(f"Regression results plot saved to {output_path}")
    plt.close(fig)


def plot_transition_results(
    curr_paths,
    next_paths,
    curr_areas,
    actions,
    true_next_areas,
    pred_next_areas,
    data_dir,
    output_dir,
    filename="transition_results.png",
    n=10,
):
    """Plots current and next images with transition details."""
    curr_paths = np.array(curr_paths)
    next_paths = np.array(next_paths)
    curr_areas = np.array(curr_areas)
    actions = np.array(actions)
    true_next_areas = np.array(true_next_areas)
    pred_next_areas = np.array(pred_next_areas)

    if len(curr_paths) > n:
        indices = np.random.permutation(len(curr_paths))[:n]
        curr_paths = curr_paths[indices]
        next_paths = next_paths[indices]
        curr_areas = curr_areas[indices]
        actions = actions[indices]
        true_next_areas = true_next_areas[indices]
        pred_next_areas = pred_next_areas[indices]

    # 2 rows, N columns
    fig, axs = plt.subplots(2, len(curr_paths), figsize=(len(curr_paths), 2))

    for i in range(len(curr_paths)):
        # Current Image
        curr_path = curr_paths[i]
        if isinstance(curr_path, (bytes, np.bytes_)):
            curr_path = curr_path.decode("utf-8")
        curr_full_path = f"{data_dir}/{curr_path}"

        ax_curr = axs[0, i]
        try:
            img = Image.open(curr_full_path)
            ax_curr.imshow(img)
            # Format action string nicely
            action_str = (
                f"({actions[i][0]:.1f}, {actions[i][1]:.1f}, {actions[i][2]:.1f})"
            )
            ax_curr.set_title(f"Area: {curr_areas[i]:.0f}\n{action_str}", fontsize=8)
        except Exception as e:
            logging.warning(f"Could not load image {curr_full_path}: {e}")
        ax_curr.axis("off")

        # Next Image
        next_path = next_paths[i]
        if isinstance(next_path, (bytes, np.bytes_)):
            next_path = next_path.decode("utf-8")
        next_full_path = f"{data_dir}/{next_path}"

        ax_next = axs[1, i]
        try:
            img = Image.open(next_full_path)
            ax_next.imshow(img)
            ax_next.set_title(
                f"True Next: {true_next_areas[i]:.0f}\nPred Next: {pred_next_areas[i]:.0f}",
                fontsize=8,
            )
        except Exception as e:
            logging.warning(f"Could not load image {next_full_path}: {e}")
        ax_next.axis("off")

    plt.tight_layout()
    output_path = output_dir / filename
    plt.savefig(output_path)
    logging.info(f"Transition results plot saved to {output_path}")
    plt.close(fig)


def plot_rollout_trajectories(
    trajectories,
    data_dir,
    output_dir,
    policy_name="policy",
    filename="rollout_trajectories.pdf",
):
    """
    Plot K rollout trajectories as a grid.

    Each row is a trajectory, each column is a timestep.

    Args:
        trajectories: List of trajectories, where each trajectory is a list of dicts
                     containing 'image_path', 'action', 'area', 'similarity'
        data_dir: Base directory for images
        output_dir: Output directory for plot
        policy_name: Name of the policy used
        filename: Output filename
    """
    n_trajectories = len(trajectories)
    n_steps = len(trajectories[0]) if trajectories else 0

    if n_trajectories == 0 or n_steps == 0:
        logging.warning("No trajectories to plot")
        return

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
                title += f"Area: {step['dataset_area']:.0f}"

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
