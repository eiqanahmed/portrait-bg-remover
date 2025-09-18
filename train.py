import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, BackupAndRestore
from tensorflow.keras import mixed_precision
from model import build_model  # ✅ make sure your model fuses to ONE channel (see note below)

# Mixed precision for T4
mixed_precision.set_global_policy("mixed_float16")

# Globals for static shapes in tf.numpy_function
image_h = 256
image_w = 256


def create_dir(path): os.makedirs(path, exist_ok=True)


def load_dataset(path, split=0.1):
    """Loading images and masks"""
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    # Splitting the data into training and testing
    split_size = int(len(X) * split)

    train_x, valid_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(Y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y)


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))
    x = x/255.0
    x = x.astype(np.float32)
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (image_w, image_h))
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    x = np.concatenate([x, x, x, x], axis=-1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([image_h, image_w, 3])
    y.set_shape([image_h, image_w, 4])
    return x, y


def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds


if __name__ == "__main__":
    import pandas as pd  # for reading CSV to set initial_epoch

    np.random.seed(42)
    tf.random.set_seed(42)

    # dirs & paths
    create_dir("files")
    create_dir("files/backup")                 # for BackupAndRestore state
    best_ckpt = os.path.join("files", "best.keras")   # best-by-val_loss
    last_ckpt = os.path.join("files", "last.keras")   # latest epoch (always overwritten)
    csv_path = os.path.join("files", "data.csv")

    # hyperparams
    image_h = 256
    image_w = 256
    input_shape = (image_h, image_w, 3)
    batch_size = 8
    lr = 1e-4
    num_epochs = 100

    #data
    dataset_path = "data/person-segmentation/people_segmentation"
    (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path, split=0.2)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)}")

    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)

    # ---------- build or resume from last checkpoint
    try:
        model = tf.keras.models.load_model(last_ckpt, compile=False)
        print(f"Loaded latest checkpoint: {last_ckpt}")
    except Exception:
        model = build_model(input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        jit_compile=False,
    )


    initial_epoch = 0
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if len(df) and "epoch" in df.columns:
                initial_epoch = int(df["epoch"].iloc[-1]) + 1
        except Exception:
            pass

    # callbacks (exact resume + keep best & latest)
    callbacks = [
        BackupAndRestore(backup_dir="files/backup"),        # exact resume: optimizer + epoch/step
        ModelCheckpoint(best_ckpt, monitor="val_loss", save_best_only=True),
        ModelCheckpoint(last_ckpt, save_best_only=False),   # always keep latest epoch
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=False),
    ]

    # ---------- train
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=num_epochs,
        initial_epoch=initial_epoch,   # ignored if BackupAndRestore finds a backup state
        callbacks=callbacks,
    )
