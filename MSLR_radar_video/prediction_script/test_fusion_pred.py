import os
import re
import h5py
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers

# ───────────────────────────────────────────────────────────────────────────────
# 1. USER‐ADJUST THESE PATHS
# ───────────────────────────────────────────────────────────────────────────────

MD_MODEL_PATH    = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/final_weights/radar_final.h5"
VID_MODEL_PATH   = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/final_weights/video_final.h5"
BASE_VAL_DIR     = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Test/Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Test/test"
PROB_MATRIX_DIR  = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/results"
PRED_DIR  = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/results/predictions"

# filenames (we will prepend PROB_MATRIX_DIR when saving)
OUT_MD_PROB_XLSX    = "prob_md_test.xlsx"
OUT_MD_LOGIT_XLSX   = "logit_md_test.xlsx"
OUT_VID_PROB_XLSX   = "prob_video_test.xlsx"
OUT_VID_LOGIT_XLSX  = "logit_video_test.xlsx"

OUT_MD_CSV     = "md_preds_test.csv"
OUT_VID_CSV    = "video_preds_test.csv"
OUT_FUSE_CSV   = "fusion_preds_test.csv"

# ───────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────

NUM_CLASSES    = 126
IMG_SIZE       = (96, 96)
FRAME_TARGET   = 64
SPATIAL_SIZE   = 96
CHANNEL_DIM    = 1
SAMPLE_PATTERN = re.compile(r"^SAMPLE_(\d+)$", re.IGNORECASE)

# ensure output directory exists
os.makedirs(PROB_MATRIX_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
# ───────────────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

def load_md_images_from_folder(sample_dir):
    pngs = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(".png")])[:3]
    imgs = []
    for fname in pngs:
        img = cv2.imread(os.path.join(sample_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load {fname}")
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        imgs.append(img[..., np.newaxis])
    while len(imgs) < 3:
        imgs.append(np.zeros((IMG_SIZE[0], IMG_SIZE[1], CHANNEL_DIM), dtype="float32"))
    return imgs

def load_video_h5_from_folder(sample_dir):
    h5s = [f for f in os.listdir(sample_dir) if f.lower().endswith(".h5")]
    if len(h5s) != 1:
        raise RuntimeError(f"{sample_dir}: expected exactly one .h5, got {h5s}")
    with h5py.File(os.path.join(sample_dir, h5s[0]), "r") as f:
        vol = f["video"][()]
    vol = vol.astype("float32")
    # normalize
    mn, mx = vol.min(), vol.max()
    vol = (vol - mn) / max(1e-8, mx - mn)
    # pad/crop frames
    if vol.shape[0] < FRAME_TARGET:
        vol = np.pad(vol, ((0, FRAME_TARGET - vol.shape[0]), (0,0), (0,0)), constant_values=0.0)
    else:
        vol = vol[:FRAME_TARGET]
    # pad/crop spatial
    h, w = vol.shape[1], vol.shape[2]
    pad_h, pad_w = max(0, SPATIAL_SIZE - h), max(0, SPATIAL_SIZE - w)
    vol = np.pad(vol, ((0,0),(0,pad_h),(0,pad_w)), constant_values=0.0)[:,
               :SPATIAL_SIZE, :SPATIAL_SIZE]
    return vol[np.newaxis, ..., np.newaxis]

def extract_id(foldername):
    m = SAMPLE_PATTERN.match(foldername)
    if not m:
        raise ValueError(f"Bad folder name {foldername}")
    return int(m.group(1))

def build_logits_model(model, prefix):
    last = model.layers[-1]
    if isinstance(last, tf.keras.layers.Activation):
        return Model(model.input, last.input, name=prefix + "_logits")
    else:
        W, b = last.get_weights()
        pen = last.input
        logits_layer = layers.Dense(last.units, activation=None, name=prefix + "_logits_dense")
        out = logits_layer(pen)
        logits_layer.set_weights([W, b])
        return Model(model.input, out, name=prefix + "_logits")

# ───────────────────────────────────────────────────────────────────────────────
# 4. LOAD MODELS
# ───────────────────────────────────────────────────────────────────────────────

print("Loading MD CNN…")
md_model = tf.keras.models.load_model(MD_MODEL_PATH, custom_objects={"K": K})
md_logits_model = build_logits_model(md_model, "md")

print("Loading Video model…")
vid_model = tf.keras.models.load_model(VID_MODEL_PATH)
vid_logits_model = build_logits_model(vid_model, "vid")

# ───────────────────────────────────────────────────────────────────────────────
# 5. INFERENCE
# ───────────────────────────────────────────────────────────────────────────────

folders = sorted([d for d in os.listdir(BASE_VAL_DIR) if SAMPLE_PATTERN.match(d)],
                 key=lambda x: int(SAMPLE_PATTERN.match(x).group(1)))
n = len(folders)
ids         = np.zeros(n, dtype="int32")
md_probs    = np.zeros((n, NUM_CLASSES), dtype="float32")
md_logits   = np.zeros((n, NUM_CLASSES), dtype="float32")
vid_probs   = np.zeros((n, NUM_CLASSES), dtype="float32")
vid_logits  = np.zeros((n, NUM_CLASSES), dtype="float32")

for i, f in enumerate(folders):
    sample_id = extract_id(f)
    ids[i] = sample_id
    d = os.path.join(BASE_VAL_DIR, f)

    # MD branch
    imgs = load_md_images_from_folder(d)
    p_md = md_model.predict([img[np.newaxis] for img in imgs], verbose=0)[0]
    z_md = md_logits_model.predict([img[np.newaxis] for img in imgs], verbose=0)[0]
    md_probs[i]  = p_md
    md_logits[i] = z_md

    # Video branch
    vol = load_video_h5_from_folder(d)
    p_vid = vid_model.predict(vol, verbose=0)[0]
    z_vid = vid_logits_model.predict(vol, verbose=0)[0]
    vid_probs[i]  = p_vid
    vid_logits[i] = z_vid

    if (i+1) % 500 == 0 or (i+1) == n:
        print(f"Processed {i+1}/{n} samples")

print("Inference complete.\n")

# ───────────────────────────────────────────────────────────────────────────────
# 6. SAVE MATRICES
# ───────────────────────────────────────────────────────────────────────────────

def save_matrix(matrix, cols, fname, to_excel=True):
    df = pd.DataFrame(matrix, columns=cols)
    df.insert(0, "id", ids)
    df = df.sort_values("id").reset_index(drop=True)
    path = os.path.join(PROB_MATRIX_DIR, fname)
    if to_excel:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Saved {path}")

columns = [f"class_{i}" for i in range(NUM_CLASSES)]
# after‐softmax probabilities
save_matrix(md_probs,   columns, OUT_MD_PROB_XLSX,   to_excel=True)
save_matrix(vid_probs,  columns, OUT_VID_PROB_XLSX,   to_excel=True)
# before‐softmax logits
save_matrix(md_logits,  columns, OUT_MD_LOGIT_XLSX,  to_excel=True)
save_matrix(vid_logits, columns, OUT_VID_LOGIT_XLSX, to_excel=True)

# ───────────────────────────────────────────────────────────────────────────────
# 7. SAVE PREDICTIONS CSV
# ───────────────────────────────────────────────────────────────────────────────

# MD‐only
md_pred = np.argmax(md_probs, axis=1)
md_df = pd.DataFrame({"id": ids, "pred": md_pred}).sort_values("id").reset_index(drop=True)
md_df.to_csv(os.path.join(PRED_DIR, OUT_MD_CSV), index=False)
print(f"Saved {OUT_MD_CSV}")

# Video‐only
vid_pred = np.argmax(vid_probs, axis=1)
vid_df = pd.DataFrame({"id": ids, "pred": vid_pred}).sort_values("id").reset_index(drop=True)
vid_df.to_csv(os.path.join(PRED_DIR, OUT_VID_CSV), index=False)
print(f"Saved {OUT_VID_CSV}")

# Fusion 50/50
fuse_probs = 0.5 * md_probs + 0.5 * vid_probs
fuse_pred  = np.argmax(fuse_probs, axis=1)
fuse_df    = pd.DataFrame({"id": ids, "pred": fuse_pred}).sort_values("id").reset_index(drop=True)
fuse_df.to_csv(os.path.join(PRED_DIR, OUT_FUSE_CSV), index=False)
print(f"Saved {OUT_FUSE_CSV}")

print("\nAll outputs written to:", PROB_MATRIX_DIR)
