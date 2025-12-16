import argparse
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""CLI tool for estimating the total value of coins in images.

The script reads a directory with images and a CSV file containing ground truth
values, detects coins using HSV masking and contours, estimates the summed
value, and computes MAE against the ground truth.
"""

def load_image(path):
    """Load an image from disk and return it in RGB format."""
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image, color=False, ax=None, title=None):
    """Display an image (RGB or grayscale) on the provided axis without axes."""
    if ax is None:
        ax = plt.gca()
    if color:
        ax.imshow(image)
    else:
        ax.imshow(image, "gray")
    ax.axis("off")
    if title:
        ax.set_title(title)

def build_mask(hsv, s_thr=80, v_thr=90):
    """Create a binary mask for coins from an HSV image (with opening/closing)."""
    h, s, v = cv2.split(hsv)
    # Keep high-saturation, high-value pixels with hue outside the green range.
    mask = (((h < 60) | (h > 170)) & (s > s_thr) & (v > v_thr)).astype("uint8")
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, h

def component_value(h_vals):
    """Assign a coin value to a contour based on its mean hue component."""
    mean_h = h_vals.mean()
    # Hue buckets tuned to distinguish coin types: red, gold, and star.
    if mean_h < 10 or mean_h > 170:
        return 2  # red
    if mean_h < 35:
        return 1  # gold
    return 5      # star


def visualize_steps(img, hsv, mask, contours, accepted, name, pred=None, truth=None):
    """Show key processing steps and the hue histogram."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    if pred is not None or truth is not None:
        fig.suptitle(f"Prediction: {pred} | Ground truth: {truth}", fontsize=14)
    display_image(img, color=True, ax=axes[0, 0], title=f"Input: {name}")
    display_image(hsv[:, :, 0], ax=axes[0, 1], title="H channel")
    display_image(mask, ax=axes[0, 2], title="Mask (morphology)")

    # Strong contrast overlay so contours are visible against the image
    overlay_all = img.copy()
    filled = img.copy()
    cv2.drawContours(overlay_all, contours, -1, (0, 255, 255), 3)  # thick outline
    cv2.drawContours(filled, contours, -1, (0, 255, 255), -1)      # filled areas
    blend = cv2.addWeighted(filled, 0.35, overlay_all, 0.65, 0)
    display_image(blend, color=True, ax=axes[1, 0], title="All contours")

    overlay_accepted = img.copy()
    cv2.drawContours(overlay_accepted, accepted, -1, (0, 255, 255), 3)
    display_image(overlay_accepted, color=True, ax=axes[1, 1], title="Filtered contours")

    hue_vals = hsv[:, :, 0][mask > 0]
    ax_hist = axes[1, 2]
    ax_hist.hist(hue_vals.ravel(), bins=180, range=(0, 180), color="steelblue")
    ax_hist.set_title("Hue histogram (mask)")
    ax_hist.set_xlim(0, 180)
    ax_hist.axvspan(0, 10, color="red", alpha=0.15, label="Red (0-10)")
    ax_hist.axvspan(170, 180, color="red", alpha=0.15, label="Red (170-180)")
    ax_hist.axvspan(10, 35, color="gold", alpha=0.15, label="Gold (10-35)")
    ax_hist.axvspan(35, 60, color="purple", alpha=0.1, label="Star (35-60)")
    ax_hist.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def estimate_value(img, show=False, name="", truth=None):
    """Estimate the total coin value in an image, optionally showing steps."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask, h = build_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    accepted = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600 or area > 3200:
            continue
        x, y, w, hb = cv2.boundingRect(cnt)
        ratio = w / hb
        if ratio < 0.7 or ratio > 1.3:
            continue
        per = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (per * per + 1e-5)
        if circularity < 0.5:
            continue
        accepted.append(cnt)
        mask_cnt = np.zeros(mask.shape, dtype="uint8")
        cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
        total += component_value(h[mask_cnt == 255])

    if show:
        visualize_steps(img, hsv, mask, contours, accepted, name, pred=total, truth=truth)
    return total


def parse_args():
    """Define CLI arguments and return parsed values."""
    parser = argparse.ArgumentParser(description="Estimate coin values with optional visualization of steps.")
    parser.add_argument("data_dir", help="Path to directory with images and coin_value_count.csv.")
    parser.add_argument("--show", action="store_true", help="Show graphical intermediate steps.")
    return parser.parse_args()


def main():
    """Run the CLI flow: load data, compute predictions, and print MAE."""
    args = parse_args()
    data_dir = args.data_dir
    csv_path = os.path.join(data_dir, "coin_value_count.csv")
    gt = pd.read_csv(csv_path)
    preds = {}
    for name in gt["image_name"]:
        img = load_image(os.path.join(data_dir, name))
        truth = int(gt.loc[gt.image_name == name, "coins_value"].iloc[0])
        preds[name] = estimate_value(img, show=args.show, name=name, truth=truth)
        print(f"{name}: pred={preds[name]}, truth={truth}")
    mae = np.mean(np.abs(gt["coins_value"] - gt["image_name"].map(preds)))
    print(f"Mean Absolute Error (MAE): {mae}")

if __name__ == "__main__":
    main()
