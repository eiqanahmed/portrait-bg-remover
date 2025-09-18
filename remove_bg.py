import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import mixed_precision


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def reduce_halo_mask(mask, erosion_size=1):
    """Simple but effective halo reduction"""
    # Convert to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Light morphological operations to clean up
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)

    # Very light erosion to reduce halo (shrink the mask slightly)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size * 2 + 1, erosion_size * 2 + 1))
    mask_eroded = cv2.erode(mask_closed, kernel_erode, iterations=1)

    # Light Gaussian blur for smooth edges
    mask_smooth = cv2.GaussianBlur(mask_eroded, (3, 3), 0.8)

    return mask_smooth.astype(np.float32) / 255.0


def create_anti_halo_transparent(original_image, mask):
    """Create transparent image with anti-halo processing"""
    # Ensure mask is single channel
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Reduce halo effect
    clean_mask = reduce_halo_mask(mask, erosion_size=1)

    # Create alpha with slight contrast boost
    alpha_enhanced = np.power(clean_mask, 0.8)  # Slight gamma correction
    alpha = np.clip((alpha_enhanced * 255).astype(np.uint8), 0, 255)

    # Convert to BGRA
    bgra = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    return bgra


def create_preview_with_background(rgba_image, bg_color=(255, 255, 255)):
    """Create a preview image with colored background"""
    # Create background
    h, w = rgba_image.shape[:2]
    background = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Extract alpha channel
    alpha = rgba_image[:, :, 3] / 255.0

    # Simple alpha blending
    result = background.copy()
    for c in range(3):
        result[:, :, c] = (
                rgba_image[:, :, c] * alpha +
                background[:, :, c] * (1 - alpha)
        ).astype(np.uint8)

    return result


def create_final_outputs(rgba_image, name, output_dir):
    """Create only the three required outputs: transparent, white bg, black bg"""
    # White background
    white_bg = create_preview_with_background(rgba_image, (255, 255, 255))
    cv2.imwrite(f"{output_dir}/{name}_white_bg.png", white_bg)

    # Black background
    black_bg = create_preview_with_background(rgba_image, (0, 0, 0))
    cv2.imwrite(f"{output_dir}/{name}_black_bg.png", black_bg)

    # Transparent background
    cv2.imwrite(f"{output_dir}/{name}_transparent.png", rgba_image)

    return white_bg, black_bg, rgba_image


# Global params
image_h = 256
image_w = 256

if __name__ == "__main__":
    # Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    # Directory for storing files
    create_dir("test/final_outputs")

    mixed_precision.set_global_policy("mixed_float16")

    # Loading model
    model = tf.keras.models.load_model("files/best.keras", compile=False)

    data_x = glob("input_images/*")

    print(f"Found {len(data_x)} images: {data_x}")

    for path in tqdm(data_x, total=len(data_x)):
        name = path.split("/")[-1].split(".")[0]

        # Load and preprocess image
        original_image = cv2.imread(path, cv2.IMREAD_COLOR)
        if original_image is None:
            print(f"Could not load image: {path}")
            continue

        h, w, _ = original_image.shape

        # Prepare input for model
        x = cv2.resize(original_image, (image_w, image_h))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        # Prediction
        y = model.predict(x, verbose=0)[0][:, :, -1]
        prob = y.astype(np.float32)

        # Resize mask back to original dimensions
        mask_resized = cv2.resize(prob, (w, h))

        # Creating anti-halo transparent image
        anti_halo_transparent = create_anti_halo_transparent(original_image, mask_resized)

        # Creating the final outputs: transparent, white background, black background
        white_bg, black_bg, transparent = create_final_outputs(
            anti_halo_transparent, name, "test/final_outputs"
        )

        # comparison image showing all three outputs
        line = np.ones((original_image.shape[0], 10, 3), dtype=np.uint8) * 128

        comparison = np.concatenate([
            white_bg, line,
            black_bg, line,
            white_bg  # Show white background twice since transparent can't be visualized directly
        ], axis=1)

        cv2.imwrite(f"test/final_outputs/{name}_comparison.png", comparison)

        print(f"Processed {name} - Created transparent, white bg, and black bg versions")