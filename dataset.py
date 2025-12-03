import numpy as np
import polars as pl
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold


def load_data(file: str):
    """Load dataframe from parquet file using polars.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        DataFrame
    """
    return pl.read_parquet(file)


def prepare_data(df: pl.DataFrame, mode: str = "decoder"):
    """
    Prepare data for training based on mode.

    Args:
        df: Input DataFrame
        mode: 'decoder' or 'transition'

    Returns:
        Prepared DataFrame
    """
    if mode == "transition":
        # Sort by experiment, zone, plant_id, time to ensure correct order
        df = df.sort(["experiment", "zone", "plant_id", "time"])

        # Create next_embedding by shifting within groups
        # We group by experiment, zone, plant_id to ensure we don't transition across plants
        df = df.with_columns(
            pl.col("embedding")
            .shift(-1)
            .over(["experiment", "zone", "plant_id"])
            .alias("next_embedding")
        )

        # Also create next_area for evaluation
        if "area" in df.columns:
            df = df.with_columns(
                pl.col("area")
                .shift(-1)
                .over(["experiment", "zone", "plant_id"])
                .alias("next_area")
            )

        # Create next_image_path by shifting within groups
        if "image_path" in df.columns:
            df = df.with_columns(
                pl.col("image_path")
                .shift(-1)
                .over(["experiment", "zone", "plant_id"])
                .alias("next_image_path")
            )

        # Filter out rows where truncated is True (for inputs) or next_embedding is null
        df = df.filter(
            ~pl.col("truncated")
            & pl.col("next_embedding").is_not_null()
            & pl.col("next_area").is_not_null()
            & pl.col("next_image_path").is_not_null()
        )

    # drop rows if area or action are null or nan
    df = df.filter(
        pl.col("area").is_not_null()
        & pl.col("area").is_not_nan()
        & pl.col("red_coef").is_not_null()
        & pl.col("red_coef").is_not_nan()
        & pl.col("white_coef").is_not_null()
        & pl.col("white_coef").is_not_nan()
        & pl.col("blue_coef").is_not_null()
        & pl.col("blue_coef").is_not_nan()
    )

    return df


def get_folds(df: pl.DataFrame, n_splits: int = 5, seed: int = 42):
    """
    Generate stratified group k-folds.

    Args:
        df: DataFrame to split
        n_splits: Number of folds
        seed: Random seed

    Yields:
        Tuple of (fold_index, train_df, val_df)
    """
    # Create groups column for splitting
    groups = (
        df.select(pl.concat_str([pl.col("experiment"), pl.col("zone")], separator="_"))
        .to_series()
        .to_list()
    )
    y = df["bolted"].fill_null(0).to_list()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # We need a dummy X for the split method
    X = np.zeros(len(y))

    for i, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_df = df[train_idx]
        val_df = df[val_idx]
        # normalize area by train_df
        factor = train_df["area"].max()
        train_df = train_df.with_columns(pl.col("area").cast(pl.Float64) / factor)
        val_df = val_df.with_columns(pl.col("area").cast(pl.Float64) / factor)

        # Also normalize next_area if it exists (for transition mode)
        if "next_area" in train_df.columns:
            train_df = train_df.with_columns(
                pl.col("next_area").cast(pl.Float64) / factor
            )
            val_df = val_df.with_columns(pl.col("next_area").cast(pl.Float64) / factor)

        yield i, train_df, val_df, factor


def convert_to_dataset(
    df: pl.DataFrame,
    input_cols: list[str],
    target_cols: list[str],
    batch_size: int,
    buffer_size: int = 1024,
    shuffle: bool = True,
    drop_remainder: bool = True,
):
    """
    Convert polars DataFrame to TensorFlow dataset.

    Args:
        df: DataFrame to convert
        input_cols: List of input columns
        target_cols: List of target columns
        batch_size: Batch size
        buffer_size: Buffer size for shuffling
        shuffle: Whether to shuffle the dataset
        drop_remainder: Whether to drop the last batch if it's smaller than batch_size

    Returns:
        TensorFlow dataset
    """
    data = {}
    for col in input_cols + target_cols:
        # Handle list columns (like embedding) by stacking
        if df[col].dtype == pl.List:
            data[col] = np.stack(df[col].to_list())
        else:
            data[col] = df[col].to_numpy()

    # Add next_area if it exists (for transition mode evaluation)
    if "next_area" in df.columns:
        data["next_area"] = df["next_area"].to_numpy()

    # Add area if it exists and is not already included (for visualization)
    if "area" in df.columns and "area" not in data:
        data["area"] = df["area"].to_numpy()

    if "image_path" in df.columns:
        data["path"] = np.array(df["image_path"].to_list())

    if "next_image_path" in df.columns:
        data["next_path"] = np.array(df["next_image_path"].to_list())

    ds = tf.data.Dataset.from_tensor_slices(data)

    if shuffle:
        ds = ds.shuffle(buffer_size)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(1)
    return ds


def print_dataset_stats(ds: tf.data.Dataset, split: str, target_cols: list[str] = None):
    """Print dataset statistics."""

    # If we have bolted target, show detailed classification stats
    if target_cols and "bolted" in target_cols:

        def reduce_fn(state, batch):
            labels = batch["bolted"]
            count = tf.shape(labels)[0]
            positives = tf.reduce_sum(labels)
            return (state[0] + count, state[1] + positives)

        total, positives = ds.reduce((0, 0), reduce_fn)
        negatives = total - positives
        pos_ratio = positives / total if total > 0 else 0

        print(f"{split} set:")
        print(f"  Total samples: {total}")
        print(f"  Positive samples: {positives}")
        print(f"  Negative samples: {negatives}")
        print(f"  Positive ratio: {pos_ratio:.2f}")
    else:
        count = 0
        for _ in ds:
            count += 1
        print(f"{split} set: {count} batches")
