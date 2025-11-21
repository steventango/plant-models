import numpy as np
import polars as pl
import tensorflow as tf


def load_data(file: str):
    """Load dataframe from parquet file using polars.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        DataFrame
    """
    return pl.read_parquet(file)


def split_data(df: pl.DataFrame, train_groups: list[tuple[int, int]]):
    """
    Split data into train and test sets based on experiment and zone.

    Args:
        df: DataFrame to split
        train_groups: List of (experiment, zone) tuples to use for training.
                      All other groups will be used for testing.

    Returns:
        Tuple of (train_df, test_df)
    """
    mask_expr = pl.lit(False)
    for exp, zone in train_groups:
        current = (pl.col("experiment") == exp) & (pl.col("zone") == zone)
        mask_expr = mask_expr | current
    train_df = df.filter(mask_expr)
    test_df = df.filter(~mask_expr)
    return train_df, test_df


def convert_to_dataset(
    df: pl.DataFrame,
    split: str,
    batch_size: int,
    train_steps: int = 0,
    buffer_size: int = 1024,
):
    """
    Convert polars DataFrame to TensorFlow dataset.

    Args:
        df: DataFrame to convert
        split: "train" or "test"
        batch_size: Batch size
        train_steps: Number of training steps
        buffer_size: Buffer size for shuffling

    Returns:
        TensorFlow dataset
    """
    embeddings = np.stack(df["embedding"].to_list())
    labels = df["bolted"].fill_null(0).cast(pl.Int32).to_numpy()
    paths = np.array(df["image_path"].to_list())

    ds = tf.data.Dataset.from_tensor_slices(
        {
            "embedding": embeddings,
            "label": labels,
            "path": paths,
        }
    )

    if split == "train":
        ds = ds.repeat().shuffle(buffer_size).take(train_steps)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
    return ds


def print_dataset_stats(ds: tf.data.Dataset, split: str):
    """Print dataset statistics."""

    def reduce_fn(state, batch):
        labels = batch["label"]

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
