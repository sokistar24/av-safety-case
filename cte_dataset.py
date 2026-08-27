"""
cte_dataset.py
Dataset loader for CTE prediction (the TaxiNet-style regression task, adapted to a car).

The model learns to predict CONTINUOUS cte from a camera image. We discretize cte
into states ONLY for the confusion-matrix / PRISM analysis, never for training — so
this file owns the single shared bin_cte() used everywhere, keeping training-eval and
the safety analysis perfectly consistent.

Data comes from collect_cte_data.py:
  data_cte/<track>/cte_log.csv   columns: image, cte, speed, steer_cmd, lap, track, mode
  data_cte/<track>/IMG/frame_XXXXXX.jpg
Paths in the CSV are relative to the track folder (we wrote them ourselves — no
Windows-path issue), so we join image -> <data_dir>/<image>.

State discretization (Framing B: lane-keeping, safe band offset from centerline;
road edge at |cte|=2.0 from the observation that beyond ~2 the car is on grass):
  0 = LANE      [-0.8, 0.0]      (normal driving band around the lane)
  1 = near-left [-1.4, -0.8)
  3 = far-left  [-2.0, -1.4)
  2 = near-right (0.0, 0.8]
  4 = far-right (0.8, 2.0]
 -1 = ERROR (off-road)  |cte| > 2.0
This mirrors TaxiNet's {0 center, 1/3 left, 2/4 right, -1 error}.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Camera is 160W x 120H. Crop sky/horizon (top rows), keep the road, then resize.
CROP_TOP = 40          # drop the top 40 rows (sky/horizon)
CROP_BOTTOM = 120      # keep down to row 120 (full bottom)
IMG_W, IMG_H = 160, 80 # after crop it's 160x80; keep that as the net input size

# ---- THE shared state discretization ----------------------------------------
ROAD_EDGE = 2.0        # |cte| beyond this = off-road error state
def bin_cte(x):
    """Map a continuous cte value to a discrete state. Single source of truth."""
    if x < -ROAD_EDGE or x > ROAD_EDGE:
        return -1                       # off-road error
    if -0.8 <= x <= 0.0:  return 0      # LANE
    if -1.4 <= x < -0.8:  return 1      # near-left
    if -2.0 <= x < -1.4:  return 3      # far-left
    if  0.0 <  x <= 0.8:  return 2      # near-right
    if  0.8 <  x <= 2.0:  return 4      # far-right
    return 0                            # (exact boundary fallback)

# ordered list of on-road + error states, for building confusion matrices
STATES = [3, 1, 0, 2, 4, -1]
STATE_NAMES = {0: "LANE", 1: "near-L", 3: "far-L", 2: "near-R", 4: "far-R", -1: "ERROR"}


def load_cte_dataframe(data_dir):
    """Load cte_log.csv and resolve image paths to absolute paths under data_dir."""
    csv_path = os.path.join(data_dir, "cte_log.csv")
    df = pd.read_csv(csv_path)
    df["image_path"] = df["image"].apply(lambda p: os.path.join(data_dir, p))
    return df


class CTEDataset(Dataset):
    def __init__(self, dataframe, augment=False, seed=42):
        self.df = dataframe.reset_index(drop=True)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[CROP_TOP:CROP_BOTTOM, :, :]       # crop sky/horizon
        img = cv2.resize(img, (IMG_W, IMG_H))
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row["image_path"])
        cte = float(row["cte"])

        if self.augment:
            # horizontal flip: mirrors the scene AND negates cte (left<->right)
            if self.rng.random() < 0.5:
                img = cv2.flip(img, 1)
                cte = -cte
            # mild brightness jitter for robustness (helps later distribution-shift)
            if self.rng.random() < 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.rng.uniform(0.6, 1.4), 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)   # HWC -> CHW
        return img, torch.tensor([cte], dtype=torch.float32)


def build_cte_datasets(data_dir, val_frac=0.2, seed=42, augment_train=True):
    """load -> split -> datasets. Returns (train_ds, val_ds, info)."""
    from sklearn.model_selection import train_test_split
    df = load_cte_dataframe(data_dir)
    train_df, val_df = train_test_split(df, test_size=val_frac, random_state=seed, shuffle=True)
    train_ds = CTEDataset(train_df, augment=augment_train, seed=seed)
    val_ds = CTEDataset(val_df, augment=False, seed=seed)
    info = {"total": len(df), "train": len(train_df), "val": len(val_df)}
    return train_ds, val_ds, info


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data_cte/generated_track"
    df = load_cte_dataframe(data_dir)
    print(f"Loaded {len(df)} rows from {data_dir}")
    # show state distribution using the shared binning
    states = df["cte"].apply(bin_cte)
    print("State distribution (shared bin_cte):")
    for s in STATES:
        n = (states == s).sum()
        print(f"  {s:+d} {STATE_NAMES[s]:8s}: {n:5d} ({n/len(df)*100:4.1f}%)")
    # sample one item
    ds = CTEDataset(df, augment=False)
    img, label = ds[0]
    print(f"Sample image tensor: {tuple(img.shape)}  (expect (3, {IMG_H}, {IMG_W}))")
    print(f"Sample cte label: {label.item():.4f}  -> state {bin_cte(label.item())}")
    print("cte_dataset.py self-test passed.")
