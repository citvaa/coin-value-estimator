# Coin Value Estimation

Python script `solution.py` estimates the total value of coins in photos using HSV segmentation and simple shape/ratio rules.

## Project structure
- `solution.py` - main processing and evaluation script.
- `data/` - sample folder with coin images and `coin_value_count.csv` containing reference totals.

## Requirements
- Python 3.10+
- Packages: `opencv-python`, `numpy`, `pandas`, `matplotlib`

Install dependencies (inside your active environment):
```bash
pip install opencv-python numpy pandas matplotlib
```

## Usage
Run the script on a directory that contains the images and CSV file:
```bash
python solution.py data
```

Optional visualization of intermediate steps (masking, contours, hue histogram):
```bash
python solution.py data --show
```

The script loads `coin_value_count.csv`, computes a prediction for each image, and prints the MAE (mean absolute error) between predictions and ground truth.

## How it works
1. Load image and convert to HSV color space.
2. Build a binary mask that keeps red/golden tones above saturation and brightness thresholds, then clean it with morphological operations.
3. Find contours and filter them by area, width/height ratio, and circularity to isolate coins.
4. For each accepted contour compute the mean Hue and assign denomination: 2 (red), 1 (gold), or 5 (slightly greenish-yellow star-like hue).
5. Sum denominations to get the total per image; visualization is shown when `--show` is enabled.

## Notes
- Color classification relies on empirical Hue thresholds (0-180 in OpenCV), so results depend on lighting conditions.
- For your own dataset, ensure the CSV has columns `image_name` and `coins_value`, and that images are RGB and readable by OpenCV.
