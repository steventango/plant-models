import argparse
import logging
import shutil
from pathlib import Path

import jax.numpy as jnp
import metrax.nnx
import numpy as np
import optax
import orbax.checkpoint
import polars as pl
import tensorflow as tf
from flax import nnx
from tqdm import tqdm

from dataset import (
    convert_to_dataset,
    get_folds,
    load_data,
    prepare_data,
    print_dataset_stats,
)
from metrax_monkey_patch import patch_metrax
from network import MLP
from plot import (
    plot_confusion_matrix,
    plot_metrics,
    plot_predictions,
    plot_regression_predictions,
    plot_regression_results,
    plot_transition_results,
)

patch_metrax()
tf.random.set_seed(0)


def loss_fn(
    model: MLP,
    rngs: nnx.Rngs,
    batch,
    input_cols: list[str],
    target_cols: list[str],
    output_heads: dict[str, int],
):
    # Construct input tensor
    inputs = [batch[col] for col in input_cols]
    processed_inputs = []
    for x in inputs:
        if x.ndim == 1:
            x = x[:, None]
        processed_inputs.append(x)

    x = jnp.concatenate(processed_inputs, axis=-1)

    preds = model(x, rngs)

    total_loss = 0.0

    for col in target_cols:
        y_true = batch[col]
        y_pred = preds[col]

        # Check if target is integer (classification) or float (regression)
        if col == "bolted" or jnp.issubdtype(
            y_true.dtype, jnp.integer
        ):  # Classification
            if not jnp.issubdtype(y_true.dtype, jnp.integer):
                y_true = y_true.astype(jnp.int32)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=y_pred, labels=y_true
            ).mean()
        else:  # Regression
            if y_true.ndim == 1:
                y_true = y_true[:, None]
            loss = optax.l2_loss(y_pred, y_true).mean()

        total_loss += loss

    return total_loss, preds


@nnx.jit(static_argnums=(5, 6, 7, 8))
def train_step(
    model: MLP,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
    input_cols: tuple,
    target_cols: tuple,
    output_heads: tuple,
    has_classification: bool,
    factor: float,
):
    """Train for a single step."""
    # Reconstruct output_heads dict
    output_heads_dict = dict(output_heads)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(
        model, rngs, batch, input_cols, target_cols, output_heads_dict
    )

    # Update metrics
    # For classification targets, also track predictions for F1
    if has_classification:
        # Find first classification target to get predictions and labels
        for col in target_cols:
            # Check if this is a classification target by trying to access it in batch
            # We need to pass both predictions and labels for F1
            class_preds = None
            class_labels = None
        class_preds = None
        class_labels = None
        for target_col in target_cols:
            y_true = batch[target_col]
            if target_col == "bolted" or jnp.issubdtype(y_true.dtype, jnp.integer):
                class_labels = y_true
                if not jnp.issubdtype(class_labels.dtype, jnp.integer):
                    class_labels = class_labels.astype(jnp.int32)
                class_preds = preds[target_col].argmax(axis=1)
                break

        if class_preds is not None:
            metrics.update(values=loss, predictions=class_preds, labels=class_labels)
        else:
            metrics.update(values=loss)
    else:
        metrics.update(values=loss)

    # For area regression, track MAE separately
    if "area" in output_heads_dict:
        area_true = batch["area"]
        area_pred = preds["area"].squeeze()
        area_mae_value = jnp.abs(area_pred * factor - area_true * factor).mean()
        metrics.area_mae.update(values=area_mae_value)

    optimizer.update(model, grads)


@nnx.jit(static_argnums=(4, 5, 6, 7))
def eval_step(
    model: MLP,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
    input_cols: tuple,
    target_cols: tuple,
    output_heads: tuple,
    has_classification: bool,
    factor: float,
):
    output_heads_dict = dict(output_heads)
    loss, preds = loss_fn(
        model, rngs, batch, input_cols, target_cols, output_heads_dict
    )

    # Update loss metric
    if has_classification:
        for target_col in target_cols:
            y_true = batch[target_col]
            if target_col == "bolted" or jnp.issubdtype(y_true.dtype, jnp.integer):
                class_labels = y_true
                if not jnp.issubdtype(class_labels.dtype, jnp.integer):
                    class_labels = class_labels.astype(jnp.int32)
                class_preds = preds[target_col].argmax(axis=1)
                metrics.update(
                    values=loss, predictions=class_preds, labels=class_labels
                )
                break
        else:
            metrics.update(values=loss)
    else:
        metrics.update(values=loss)

    # For area regression, track MAE separately
    if "area" in dict(output_heads):
        area_true = batch["area"]
        area_pred = preds["area"].squeeze()
        area_mae_value = jnp.abs(area_pred * factor - area_true * factor).mean()
        metrics.area_mae.update(values=area_mae_value)

    return preds


def preprocess_batch(batch):
    return {k: v for k, v in batch.items() if k not in ["path", "next_path"]}


def print_metrics(metrics: dict[str, float]):
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.2f}")


def train_and_evaluate(
    train_ds: tf.data.Dataset,
    input_cols: list[str],
    target_cols: list[str],
    output_heads: dict[str, int],
    input_dim: int,
    test_ds: tf.data.Dataset | None = None,
    decoder_model: MLP | None = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    seed: int = 0,
    description: str = "Training",
    factor: float = 1.0,
):
    model = MLP(input_dim=input_dim, output_heads=output_heads, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)

    # Add F1 metric if we have classification targets (bolted)
    has_classification = "bolted" in target_cols
    has_area_regression = "area" in target_cols

    metrics_dict = {"loss": metrax.nnx.Average()}
    if has_classification:
        metrics_dict["f1"] = metrax.nnx.FBetaScore()
    if has_area_regression:
        metrics_dict["area_mae"] = metrax.nnx.Average()

    metrics = nnx.MultiMetric(**metrics_dict)
    rngs = nnx.Rngs(seed)

    metrics_history = {
        "train_loss": [],
        "test_loss": [],
    }
    if has_classification:
        metrics_history["train_f1"] = []
        metrics_history["test_f1"] = []
    if has_area_regression:
        metrics_history["train_area_mae"] = []
        metrics_history["test_area_mae"] = []
    if decoder_model is not None:
        # Transition model with decoder evaluation
        if "test_area_mae" not in metrics_history:
            metrics_history["test_area_mae"] = []

    last_test_preds = []
    last_test_labels = []  # This might need to be a dict or list of dicts
    last_test_paths = []
    last_test_area_preds = []
    last_test_area_labels = []

    # For transition visualization
    last_test_next_paths = []
    last_test_curr_areas = []
    last_test_actions = []
    last_test_next_area_preds = []
    last_test_next_area_labels = []

    # Convert to tuples for JIT static args
    input_cols_tuple = tuple(input_cols)
    target_cols_tuple = tuple(target_cols)
    # output_heads is a dict, convert to tuple of items
    output_heads_tuple = tuple(sorted(output_heads.items()))

    pbar = tqdm(range(epochs), desc=description, unit="epoch")
    for epoch in pbar:
        for step, batch in enumerate(train_ds.as_numpy_iterator()):
            model.train()
            batch = preprocess_batch(batch)
            train_step(
                model,
                optimizer,
                metrics,
                rngs,
                batch,
                input_cols_tuple,
                target_cols_tuple,
                output_heads_tuple,
                has_classification,
                factor,
            )

        for metric, value in metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value)
        metrics.reset()

        if test_ds is not None:
            model.eval()
            for test_batch in test_ds.as_numpy_iterator():
                test_batch_paths = test_batch.get("path", [])
                test_batch = preprocess_batch(test_batch)
                preds = eval_step(
                    model,
                    metrics,
                    rngs,
                    test_batch,
                    input_cols_tuple,
                    target_cols_tuple,
                    output_heads_tuple,
                    has_classification,
                    factor,
                )

                if epoch == epochs - 1:
                    # Collect predictions for visualization (for bolted targets)
                    for target_col in target_cols:
                        if target_col == "bolted":
                            bolted_preds = preds[target_col].argmax(axis=1)
                            bolted_labels = test_batch[target_col]
                            last_test_preds.extend(bolted_preds)
                            last_test_labels.extend(bolted_labels)
                        elif target_col == "area":
                            area_preds = preds[target_col].squeeze()
                            area_labels = test_batch[target_col]
                            # Handle potential scalar vs array issues if batch size is 1
                            if area_preds.ndim == 0:
                                area_preds = jnp.array([area_preds])
                            if area_labels.ndim == 0:
                                area_labels = jnp.array([area_labels])

                            # Apply factor to get back to original scale
                            last_test_area_preds.extend(
                                [float(p * factor) for p in area_preds]
                            )
                            last_test_area_labels.extend(
                                [float(label_val * factor) for label_val in area_labels]
                            )

                    if not (
                        decoder_model is not None and "next_embedding" in target_cols
                    ):
                        if len(test_batch_paths) > 0:
                            last_test_paths.extend(test_batch_paths)

            # Compute area MAE if decoder model is provided (for transition mode)
            if decoder_model is not None and "next_embedding" in target_cols:
                area_errors = []
                for test_batch in test_ds.as_numpy_iterator():
                    test_batch_paths = test_batch.get("path", [])
                    test_batch_next_paths = test_batch.get("next_path", [])

                    # Save area and actions before preprocessing (they'll be removed)
                    test_batch_area = test_batch.get("area", None)
                    test_batch_red = test_batch.get("red_coef", None)
                    test_batch_white = test_batch.get("white_coef", None)
                    test_batch_blue = test_batch.get("blue_coef", None)

                    test_batch = preprocess_batch(test_batch)

                    # Get predicted next embeddings from transition model
                    inputs = [test_batch[col] for col in input_cols_tuple]
                    processed_inputs = []
                    for x in inputs:
                        if x.ndim == 1:
                            x = x[:, None]
                        processed_inputs.append(x)
                    x_input = jnp.concatenate(processed_inputs, axis=-1)

                    decoder_model.eval()
                    model.eval()

                    # Predict next embedding
                    next_emb_pred = model(x_input, rngs)["next_embedding"]

                    # Predict area from next embedding
                    area_pred = decoder_model(next_emb_pred, rngs)["area"].squeeze()

                    # Get actual next area
                    if "next_area" in test_batch:
                        area_true = test_batch["next_area"]
                        mae = jnp.abs(area_pred - area_true).mean()
                        area_errors.append(mae)

                        if epoch == epochs - 1:
                            # Collect data for transition visualization
                            if len(test_batch_paths) > 0:
                                last_test_paths.extend(test_batch_paths)
                            if len(test_batch_next_paths) > 0:
                                last_test_next_paths.extend(test_batch_next_paths)

                            # Current area (saved before preprocessing)
                            if test_batch_area is not None:
                                last_test_curr_areas.extend(
                                    [float(a * factor) for a in test_batch_area]
                                )

                            # Actions (saved before preprocessing)
                            if test_batch_red is not None:
                                batch_actions = []
                                for i in range(len(area_true)):
                                    r = test_batch_red[i]
                                    w = test_batch_white[i]
                                    b = test_batch_blue[i]
                                    batch_actions.append([float(r), float(w), float(b)])
                                last_test_actions.extend(batch_actions)

                            last_test_next_area_preds.extend(
                                [float(p) for p in area_pred]
                            )
                            last_test_next_area_labels.extend(
                                [float(label_val * factor) for label_val in area_true]
                            )

                if area_errors:
                    avg_area_mae = np.mean(area_errors)
                    metrics_history["test_area_mae"].append(avg_area_mae)

            for metric, value in metrics.compute().items():
                metrics_history[f"test_{metric}"].append(value)
            metrics.reset()

        # Update progress bar
        postfix = {}
        if test_ds is not None:
            if has_classification:
                postfix["test_f1"] = f"{metrics_history['test_f1'][-1]:.2f}"
            postfix["test_loss"] = f"{metrics_history['test_loss'][-1]:.2f}"
            if has_area_regression and metrics_history.get("test_area_mae"):
                postfix["area_mae"] = f"{metrics_history['test_area_mae'][-1]:.2f}"
            if decoder_model is not None and "test_area_mae" in metrics_history:
                if metrics_history["test_area_mae"]:
                    postfix["area_mae"] = f"{metrics_history['test_area_mae'][-1]:.2f}"
        else:
            if has_classification:
                postfix["train_f1"] = f"{metrics_history['train_f1'][-1]:.2f}"
            if has_area_regression and metrics_history.get("train_area_mae"):
                postfix["area_mae"] = f"{metrics_history['train_area_mae'][-1]:.2f}"
            postfix["train_loss"] = f"{metrics_history['train_loss'][-1]:.2f}"

        pbar.set_postfix(postfix)

    return (
        model,
        metrics_history,
        last_test_preds,
        last_test_labels,
        last_test_paths,
        last_test_area_preds,
        last_test_area_labels,
        last_test_next_paths,
        last_test_curr_areas,
        last_test_actions,
        last_test_next_area_preds,
        last_test_next_area_labels,
    )


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
    parser.add_argument(
        "--mode",
        type=str,
        default="decoder",
        choices=["decoder", "transition"],
        help="Training mode",
    )
    parser.add_argument(
        "--input_cols",
        type=str,
        nargs="+",
        default=["embedding"],
        help="Input columns",
    )
    parser.add_argument(
        "--target_cols",
        type=str,
        nargs="+",
        default=None,
        help="Target columns",
    )
    parser.add_argument(
        "--action_cols",
        type=str,
        nargs="+",
        default=["red_coef", "white_coef", "blue_coef"],
        help="Action columns (for transition mode)",
    )
    parser.add_argument(
        "--decoder_checkpoint",
        type=str,
        default=None,
        help="Path to decoder checkpoint for evaluating transition model (transition mode only)",
    )

    args = parser.parse_args()

    # Configure defaults based on mode
    if args.mode == "decoder":
        if args.target_cols is None:
            args.target_cols = ["bolted", "area"]
    elif args.mode == "transition":
        if args.target_cols is None:
            args.target_cols = ["next_embedding"]
        # For transition, inputs are embedding + actions
        args.input_cols = ["embedding"] + args.action_cols

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
    df = load_data(
        str(data_dir / "cleaned_offline_dataset_daily_continuous_v15.parquet")
    )
    df = prepare_data(df, mode=args.mode)
    if "bolted" in df.columns:
        df = df.with_columns(pl.col("bolted").fill_null(0).cast(pl.Int32))

    # Load decoder model if provided (for transition mode evaluation)
    decoder_model = None
    if args.decoder_checkpoint is not None and args.mode == "transition":
        logging.info(f"Loading decoder checkpoint from {args.decoder_checkpoint}")
        # Load decoder model (embedding -> area)
        decoder_input_dim = 768  # embedding dimension
        decoder_output_heads = {"area": 1}

        # Create model with dummy initialization
        temp_model = MLP(
            input_dim=decoder_input_dim,
            output_heads=decoder_output_heads,
            rngs=nnx.Rngs(0),
        )

        # Split to get graphdef and state
        graphdef, _ = nnx.split(temp_model)

        # Load checkpoint state
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_path = Path(args.decoder_checkpoint).resolve()
        ckpt = orbax_checkpointer.restore(checkpoint_path)

        # Merge graphdef with loaded state
        decoder_model = nnx.merge(graphdef, ckpt["model"])
        logging.info("Decoder model loaded successfully")

    # Determine dimensions
    input_dim = 0
    for col in args.input_cols:
        if df[col].dtype == pl.List:
            input_dim += len(df[col][0])
        else:
            input_dim += 1

    output_heads = {}
    for col in args.target_cols:
        if col == "bolted":
            output_heads[col] = 2
        elif col == "next_embedding" or col == "embedding":
            output_heads[col] = 768
        else:
            # Assume regression scalar
            output_heads[col] = 1

    logging.info(f"Mode: {args.mode}")
    logging.info(f"Input columns: {args.input_cols} (dim={input_dim})")
    logging.info(f"Target columns: {args.target_cols}")
    logging.info(f"Output heads: {output_heads}")

    # Cross Validation
    n_splits = args.n_splits
    num_epochs = args.epochs
    batch_size = 32
    cv_results = []

    # Initialize lists to collect all predictions and labels across folds for CV plotting
    cv_all_preds = []
    cv_all_labels = []
    cv_all_paths = []
    cv_all_area_preds = []
    cv_all_area_labels = []

    # Transition visualization lists
    cv_all_next_paths = []
    cv_all_curr_areas = []
    cv_all_actions = []
    cv_all_next_area_preds = []
    cv_all_next_area_labels = []

    logging.info(f"Starting {n_splits}-Fold Cross Validation...")

    for i, train_df, val_df, factor in get_folds(df, n_splits=n_splits):
        logging.info(f"\nFold {i + 1}/{n_splits}")
        train_ds = convert_to_dataset(
            train_df,
            args.input_cols,
            args.target_cols,
            batch_size,
            shuffle=True,
            drop_remainder=True,
        )
        val_ds = convert_to_dataset(
            val_df,
            args.input_cols,
            args.target_cols,
            batch_size,
            shuffle=False,
            drop_remainder=False,
        )

        print_dataset_stats(train_ds, "train")
        print_dataset_stats(val_ds, "val")

        (
            model,
            history,
            fold_preds,
            fold_labels,
            fold_paths,
            fold_area_preds,
            fold_area_labels,
            fold_next_paths,
            fold_curr_areas,
            fold_actions,
            fold_next_area_preds,
            fold_next_area_labels,
        ) = train_and_evaluate(
            train_ds,
            args.input_cols,
            args.target_cols,
            output_heads,
            input_dim,
            val_ds,
            decoder_model,
            epochs=num_epochs,
            description=f"Fold {i + 1}",
            factor=factor,
        )
        cv_results.append(history)
        cv_all_preds.extend(fold_preds)
        cv_all_labels.extend(fold_labels)
        cv_all_paths.extend(fold_paths)
        cv_all_area_preds.extend(fold_area_preds)
        cv_all_area_labels.extend(fold_area_labels)

        cv_all_next_paths.extend(fold_next_paths)
        cv_all_curr_areas.extend(fold_curr_areas)
        cv_all_actions.extend(fold_actions)
        cv_all_next_area_preds.extend(fold_next_area_preds)
        cv_all_next_area_labels.extend(fold_next_area_labels)

    logging.info("\n=== Cross Validation Summary ===")
    metrics_to_report = ["test_loss"]
    if "bolted" in args.target_cols:
        metrics_to_report.append("test_f1")
    if "area" in args.target_cols or decoder_model is not None:
        metrics_to_report.append("test_area_mae")

    for metric in metrics_to_report:
        # Get the final value for each fold
        values = [h[metric][-1] for h in cv_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        logging.info(f"Mean {metric}: {mean_val:.4f} ± {std_val:.4f}")

    plot_metrics(cv_results, "cv_metrics.png", output_dir)

    # Plot confusion matrix and predictions for bolted targets
    if "bolted" in args.target_cols and cv_all_preds and cv_all_labels:
        plot_confusion_matrix(cv_all_labels, cv_all_preds, output_dir)
        plot_predictions(
            cv_all_preds, cv_all_labels, cv_all_paths, data_dir, output_dir
        )

    # Plot regression predictions for area
    if "area" in args.target_cols and cv_all_area_preds and cv_all_area_labels:
        plot_regression_predictions(cv_all_area_preds, cv_all_area_labels, output_dir)
        if cv_all_paths:
            plot_regression_results(
                cv_all_paths,
                cv_all_area_labels,
                cv_all_area_preds,
                data_dir,
                output_dir,
            )

    # Plot transition results
    if decoder_model is not None and cv_all_next_paths:
        plot_transition_results(
            cv_all_paths,  # Note: these are accumulated in the same order
            cv_all_next_paths,
            cv_all_curr_areas,
            cv_all_actions,
            cv_all_next_area_labels,
            cv_all_next_area_preds,
            data_dir,
            output_dir,
        )

    # Save dataset with CV predictions
    if (
        ("bolted" in args.target_cols and cv_all_preds)
        or ("area" in args.target_cols and cv_all_area_preds)
    ) and cv_all_paths:
        # Decode paths if they are bytes
        decoded_paths = []
        for p in cv_all_paths:
            if isinstance(p, (bytes, np.bytes_)):
                decoded_paths.append(p.decode("utf-8"))
            else:
                decoded_paths.append(str(p))

        pred_data = {"image_path": decoded_paths}
        if "bolted" in args.target_cols and cv_all_preds:
            pred_data["bolted_pred"] = np.array(cv_all_preds)
        if "area" in args.target_cols and cv_all_area_preds:
            pred_data["area_pred"] = np.array(cv_all_area_preds)

        pred_df = pl.DataFrame(pred_data)

        # Join with the original dataframe
        df_with_preds = df.join(pred_df, on="image_path", how="left")

        # Save the dataset with predictions
        output_path = output_dir / "dataset_with_predictions.parquet"
        df_with_preds.write_parquet(output_path)
        logging.info(f"Saved dataset with CV predictions to {output_path}")

    logging.info("\nTraining Final Model on Full Dataset...")
    full_ds = convert_to_dataset(
        df,
        args.input_cols,
        args.target_cols,
        batch_size=32,
        shuffle=True,
        drop_remainder=True,
    )
    (
        final_model,
        final_history,
        final_preds,
        final_labels,
        final_paths,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = train_and_evaluate(
        full_ds,
        args.input_cols,
        args.target_cols,
        output_heads,
        input_dim,
        None,
        decoder_model,
        epochs=num_epochs,
        description="Final Model",
    )
    plot_metrics([final_history], "final_metrics.png", output_dir)
    save_model(final_model, output_dir)


if __name__ == "__main__":
    main()
