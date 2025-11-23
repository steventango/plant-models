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
        yield i, train_df, val_df


def convert_to_dataset(
    df: pl.DataFrame,
    batch_size: int,
    buffer_size: int = 1024,
):
    """
    Convert polars DataFrame to TensorFlow dataset.

    Args:
        df: DataFrame to convert
        batch_size: Batch size
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

    ds = ds.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(1)
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
