import os
import cv2
import numpy as np
import h5py

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
# Base “train” directory containing class folders (e.g., “0_A”, “1_B”, …)
BASE_TRAIN_DIR = r"C:\Users\sb3682.ECE-2V7QHQ3\My Stuff\kaggle_comp\Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Train Val\train"

# Target dimensions
FRAME_TARGET = 64    # number of frames
SPATIAL_SIZE  = 96   # height and width

# ───────────────────────────────────────────────────────────────────────────────
# TEMPORAL INTERPOLATION UTILITY
# ───────────────────────────────────────────────────────────────────────────────
def temporal_resample(frames: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Given frames as a NumPy array of shape (N, H, W), resample to exactly
    `target_frames` frames via linear interpolation along the time axis.
    Returns an array of shape (target_frames, H, W).
    """
    num_orig = frames.shape[0]
    if num_orig == 0:
        raise ValueError("Received empty frame array for interpolation.")

    if num_orig == target_frames:
        return frames.copy()

    new_idx = np.linspace(0, num_orig - 1, num=target_frames)
    H, W = frames.shape[1], frames.shape[2]
    resampled = np.zeros((target_frames, H, W), dtype=np.float32)

    for i, idx in enumerate(new_idx):
        low = int(np.floor(idx))
        high = int(np.ceil(idx))
        alpha = idx - low
        if high >= num_orig:
            high = num_orig - 1

        if low == high:
            resampled[i] = frames[low]
        else:
            resampled[i] = (1.0 - alpha) * frames[low] + alpha * frames[high]

    return resampled


# ───────────────────────────────────────────────────────────────────────────────
# PROCESS AND SAVE EACH VIDEO AS H5
# ───────────────────────────────────────────────────────────────────────────────
for class_folder in sorted(os.listdir(BASE_TRAIN_DIR)):
    class_path = os.path.join(BASE_TRAIN_DIR, class_folder)
    if not os.path.isdir(class_path):
        continue

    for sample_folder in sorted(os.listdir(class_path)):
        if not sample_folder.startswith("SAMPLE_"):
            continue
        sample_path = os.path.join(class_path, sample_folder)
        if not os.path.isdir(sample_path):
            continue

        # Locate the .mkv file in this sample folder
        video_file = None
        for fname in os.listdir(sample_path):
            if fname.lower().endswith(".mkv"):
                video_file = os.path.join(sample_path, fname)
                break
        if video_file is None:
            print(f"Skipping {sample_path}: no .mkv found.")
            continue

        # Read all frames as grayscale
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Warning: cannot open video {video_file}. Skipping.")
            continue

        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            raw_frames.append(gray)
        cap.release()

        if len(raw_frames) == 0:
            print(f"Warning: no frames extracted from {video_file}. Skipping.")
            continue

        raw_frames = np.stack(raw_frames, axis=0)  # shape: (N, H_orig, W_orig)

        # Temporal resample to FRAME_TARGET frames
        frames_resamp = temporal_resample(raw_frames, FRAME_TARGET)  # (64, H_orig, W_orig)

        # Spatial resize each frame to SPATIAL_SIZE × SPATIAL_SIZE
        resized = np.zeros((FRAME_TARGET, SPATIAL_SIZE, SPATIAL_SIZE), dtype=np.float32)
        for i in range(FRAME_TARGET):
            resized[i] = cv2.resize(
                frames_resamp[i],
                (SPATIAL_SIZE, SPATIAL_SIZE),
                interpolation=cv2.INTER_LINEAR
            )

        # Final array shape: (64, 64, 64)
        video_array = resized  # dtype=float32, values in [0,1]

        # Save as .h5 in the same sample folder, using the same base name
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        h5_filename = f"{base_name}.h5"
        h5_path = os.path.join(sample_path, h5_filename)

        with h5py.File(h5_path, "w") as h5f:
            # Create a dataset named "video" storing the 3D array
            h5f.create_dataset(
                "video",
                data=video_array,
                compression="gzip"
            )

        print(f"Saved: {h5_path}")
