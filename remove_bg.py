import os

# Release asset for public repo (override with MODEL_URL env var if needed)
RELEASE_MODEL_URL = "https://github.com/eiqanahmed/image_background_remover/releases/download/weights-v1/best.keras.tar.gz"
LOCAL_MODEL_PATH = "files/best.keras"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import mixed_precision

import tarfile
import urllib.request


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def download_and_extract(url: str, target_dir: str, target_file: str):
    """Download a .tar.gz from GitHub Releases and extract to target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    tmp_tar = target_file + ".tar.gz"

    print(f"[weights] Downloading model from:\n{url}")
    try:
        urllib.request.urlretrieve(url, tmp_tar)
    except Exception as e:
        raise RuntimeError(f"Failed to download model from {url}: {e}")

    print("[weights] Extracting...")
    try:
        with tarfile.open(tmp_tar, "r:gz") as tar:
            
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

            def safe_extract(tarobj, path=".", members=None, *, numeric_owner=False):
                for member in tarobj.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tarobj.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar, target_dir)
    finally:
        if os.path.exists(tmp_tar):
            os.remove(tmp_tar)

    if not os.path.exists(target_file):
        raise FileNotFoundError(
            f"Model was extracted, but '{target_file}' not found in {target_dir}."
        )
    print("[weights] Ready.")


def ensure_model(local_model_path: str = LOCAL_MODEL_PATH):
    """Ensure weights exist locally; fetch from release if missing."""
    if os.path.exists(local_model_path):
        return local_model_path

    url = os.environ.get("MODEL_URL", RELEASE_MODEL_URL)
    target_dir = os.path.dirname(local_model_path) or "."
    download_and_extract(url, target_dir, local_model_path)
    return local_model_path


def reduce_halo_mask(mask, erosion_size=1):
    """Simple but effective halo reduction"""
    mask_uint8 = (mask * 255).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size * 2 + 1, erosion_size * 2 + 1))
    mask_eroded = cv2.erode(mask_closed, kernel_erode, iterations=1)

    mask_smooth = cv2.GaussianBlur(mask_eroded, (3, 3), 0.8)

    return mask_smooth.astype(np.float32) / 255.0


def create_anti_halo_transparent(original_image, mask):
    """Create transparent image with anti-halo processing"""
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    clean_mask = reduce_halo_mask(mask, erosion_size=1)

    alpha_enhanced = np.power(clean_mask, 0.8)  # Slight gamma correction
    alpha = np.clip((alpha_enhanced * 255).astype(np.uint8), 0, 255)

    bgra = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    return bgra


def create_preview_with_background(rgba_image, bg_color=(255, 255, 255)):
    """Create a preview image with colored background"""
    h, w = rgba_image.shape[:2]
    background = np.full((h, w, 3), bg_color, dtype=np.uint8)

    alpha = rgba_image[:, :, 3] / 255.0

    result = background.copy()
    for c in range(3):
        result[:, :, c] = (
            rgba_image[:, :, c] * alpha +
            background[:, :, c] * (1 - alpha)
        ).astype(np.uint8)

    return result


def create_final_outputs(rgba_image, name, output_dir):
    """Create only the three required outputs: transparent, white bg, black bg"""
    white_bg = create_preview_with_background(rgba_image, (255, 255, 255))
    cv2.imwrite(f"{output_dir}/{name}_white_bg.png", white_bg)

    black_bg = create_preview_with_background(rgba_image, (0, 0, 0))
    cv2.imwrite(f"{output_dir}/{name}_black_bg.png", black_bg)

    cv2.imwrite(f"{output_dir}/{name}_transparent.png", rgba_image)

    return white_bg, black_bg, rgba_image


# Global params
image_h = 256
image_w = 256

if __name__ == "__main__":
    
    np.random.seed(42)
    tf.random.set_seed(42)

    # Output dir
    create_dir("test/final_outputs")

    mixed_precision.set_global_policy("mixed_float16")

    # Ensure model exists locally (download from release if needed)
    ensure_model(LOCAL_MODEL_PATH)

    # Load model
    model = tf.keras.models.load_model(LOCAL_MODEL_PATH, compile=False)

    data_x = glob("input_images/*")
    print(f"Found {len(data_x)} images: {data_x}")

    for path in tqdm(data_x, total=len(data_x)):
        name = path.split("/")[-1].split(".")[0]

        original_image = cv2.imread(path, cv2.IMREAD_COLOR)
        if original_image is None:
            print(f"Could not load image: {path}")
            continue

        h, w, _ = original_image.shape

        x = cv2.resize(original_image, (image_w, image_h))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        y = model.predict(x, verbose=0)[0][:, :, -1]
        prob = y.astype(np.float32)

        mask_resized = cv2.resize(prob, (w, h))

        anti_halo_transparent = create_anti_halo_transparent(original_image, mask_resized)

        white_bg, black_bg, transparent = create_final_outputs(
            anti_halo_transparent, name, "test/final_outputs"
        )

        line = np.ones((original_image.shape[0], 10, 3), dtype=np.uint8) * 128

        comparison = np.concatenate([
            white_bg, line,
            black_bg, line,
            white_bg  # Show white background twice since transparent can't be visualized directly
        ], axis=1)

        cv2.imwrite(f"test/final_outputs/{name}_comparison.png", comparison)

        print(f"Processed {name} - Created transparent, white bg, and black bg versions")
