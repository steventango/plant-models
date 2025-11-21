import tensorflow as tf
import tensorflow_datasets as tfds


def load_and_preprocess_mnist(batch_size: int, train_steps: int):
    train_ds: tf.data.Dataset = tfds.load("mnist", split="train")
    test_ds: tf.data.Dataset = tfds.load("mnist", split="test")

    train_ds = train_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255,
            "label": sample["label"],
        }
    )
    test_ds = test_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255,
            "label": sample["label"],
        }
    )

    train_ds = train_ds.repeat().shuffle(1024)
    train_ds = (
        train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    )
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    return train_ds, test_ds
