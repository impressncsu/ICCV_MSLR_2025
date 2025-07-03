# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 12:36:15 2025

@author: sb3682
"""

import os
import cv2
import h5py
import numpy as np

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────

DIR = "Test" # change it to Test or Val depending on which data we are processing

# Path to the validation folder containing subfolders SAMPLE_<num>
VAL_DIR = r"C:\Users\sb3682.ECE-2V7QHQ3\My Stuff\kaggle_comp\Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Train Val\val"
Test_DIR      = r'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Test/Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Test/test'

if DIR == "Test":
    Main_DIR = Test_DIR
else:
    Main_DIR = VAL_DIR

# Number of frames to sample temporally and spatial dimensions to resize
FRAME_TARGET = 64
SPATIAL_SIZE = 96

# ───────────────────────────────────────────────────────────────────────────────
# FUNCTION: Temporal resampling to a fixed number of frames
# ───────────────────────────────────────────────────────────────────────────────

def temporal_resample(frames: np.ndarray, target_frames: int = FRAME_TARGET) -> np.ndarray:
    """
    Given a numpy array `frames` of shape (N, H, W), linearly resample along the
    first axis (time) to exactly `target_frames` frames. Returns an array of shape
    (target_frames, H, W), dtype float32.
    """
    N, H, W = frames.shape
    if N == target_frames:
        return frames.astype(np.float32)

    # New time indices uniformly spaced between 0 and N-1 (inclusive)
    new_idx = np.linspace(0, N - 1, num=target_frames, dtype=np.float32)
    resampled = np.zeros((target_frames, H, W), dtype=np.float32)

    for i, idx in enumerate(new_idx):
        lo = int(np.floor(idx))
        hi = min(int(np.ceil(idx)), N - 1)
        frac = idx - lo
        if lo == hi:
            resampled[i] = frames[lo]
        else:
            resampled[i] = (1.0 - frac) * frames[lo] + frac * frames[hi]

    return resampled


# ───────────────────────────────────────────────────────────────────────────────
# FUNCTION: Process a single SAMPLE_<num> folder
# ───────────────────────────────────────────────────────────────────────────────

def process_sample_folder(sample_path: str):
    """
    In `sample_path`, finds the single .mkv file, loads all frames as grayscale,
    normalizes to [0,1], resamples to FRAME_TARGET frames, resizes each frame to
    (SPATIAL_SIZE, SPATIAL_SIZE), and writes an HDF5 file named <base>.h5 in the
    same folder (dataset name 'video').
    """
    # 1. Locate the .mkv file
    mkv_files = [f for f in os.listdir(sample_path) if f.lower().endswith(".mkv")]
    if not mkv_files:
        print(f"  [!] No .mkv found in {sample_path}. Skipping.")
        return
    mkv_fname = mkv_files[0]
    mkv_path = os.path.join(sample_path, mkv_fname)

    # 2. Read all frames with OpenCV, convert to grayscale, normalize to [0,1]
    cap = cv2.VideoCapture(mkv_path)
    if not cap.isOpened():
        print(f"  [!] Cannot open {mkv_path}. Skipping.")
        return

    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0  # normalize
        raw_frames.append(gray)
    cap.release()

    if len(raw_frames) == 0:
        print(f"  [!] No frames extracted from {mkv_path}. Skipping.")
        return

    frames_np = np.stack(raw_frames, axis=0)  # shape: (N, H_orig, W_orig)

    # 3. Temporally resample to exactly FRAME_TARGET frames
    frames_resamp = temporal_resample(frames_np, target_frames=FRAME_TARGET)
    # frames_resamp shape: (FRAME_TARGET, H_orig, W_orig)

    # 4. Spatially resize each frame to (SPATIAL_SIZE, SPATIAL_SIZE)
    resized_vol = np.zeros((FRAME_TARGET, SPATIAL_SIZE, SPATIAL_SIZE), dtype=np.float32)
    for i in range(FRAME_TARGET):
        resized_vol[i] = cv2.resize(
            frames_resamp[i],
            (SPATIAL_SIZE, SPATIAL_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

    # 5. Save as HDF5 in the same folder, naming <basename>.h5
    base_name = os.path.splitext(mkv_fname)[0]
    h5_path = os.path.join(sample_path, base_name + ".h5")
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("video", data=resized_vol, compression="gzip")
    print(f"  [+] Saved {h5_path} (shape {resized_vol.shape})")


# ───────────────────────────────────────────────────────────────────────────────
# MAIN: Iterate over all SAMPLE_<num> folders in the validation directory
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Processing validation folder: {Main_DIR}")
    entries = sorted(os.listdir(Main_DIR))
    for entry in entries:
        sample_dir = os.path.join(Main_DIR, entry)
        if not os.path.isdir(sample_dir):
            continue
        if not entry.startswith("SAMPLE_"):
            continue

        print(f"Processing {entry} ...")
        process_sample_folder(sample_dir)

    print("All done.")
