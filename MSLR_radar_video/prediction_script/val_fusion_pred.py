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
# 1. === USER‐ADJUST THESE PATHS ===
# ───────────────────────────────────────────────────────────────────────────────

MD_MODEL_PATH    = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/final_weights/radar_final.h5"
VID_MODEL_PATH   = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/final_weights/video_final.h5"
BASE_VAL_DIR     = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Train Val/val"
PROB_MATRIX_DIR  = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/results"
PRED_DIR  = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/results/predictions"

# make sure output folder exists
os.makedirs(PROB_MATRIX_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
# ───────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────

NUM_CLASSES   = 126
IMG_SIZE      = (96, 96)
FRAME_TARGET  = 64
SPATIAL_SIZE  = 96
CHANNEL_DIM   = 1
SAMPLE_PATTERN = re.compile(r"^SAMPLE_(\d+)$", re.IGNORECASE)

# ───────────────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

def load_md_images_from_folder(sample_dir):
    pngs = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(".png")])[:3]
    imgs = []
    for fname in pngs:
        img = cv2.imread(os.path.join(sample_dir, fname), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        imgs.append(img[..., np.newaxis])
    while len(imgs) < 3:
        imgs.append(np.zeros((IMG_SIZE[0], IMG_SIZE[1], CHANNEL_DIM), dtype="float32"))
    return imgs

def load_video_h5_from_folder(sample_dir):
    h5s = [f for f in os.listdir(sample_dir) if f.lower().endswith(".h5")]
    if len(h5s) != 1:
        raise RuntimeError(f"Expected one .h5 in {sample_dir}, got {h5s}")
    with h5py.File(os.path.join(sample_dir, h5s[0]), "r") as f:
        vol = f["video"][()]
    vol = vol.astype("float32")
    # normalize
    mn, mx = vol.min(), vol.max()
    vol = (vol - mn) / max(1e-8, mx - mn)
    # pad/crop frames
    if vol.shape[0] < FRAME_TARGET:
        vol = np.pad(vol, ((0, FRAME_TARGET-vol.shape[0]), (0,0), (0,0)), constant_values=0.0)
    else:
        vol = vol[:FRAME_TARGET]
    # pad/crop spatial
    h, w = vol.shape[1], vol.shape[2]
    pad_h, pad_w = max(0, SPATIAL_SIZE-h), max(0, SPATIAL_SIZE-w)
    vol = np.pad(vol, ((0,0),(0,pad_h),(0,pad_w)), constant_values=0.0)[:, :SPATIAL_SIZE, :SPATIAL_SIZE]
    return vol[..., np.newaxis][np.newaxis, ...]

def extract_id(foldername):
    m = SAMPLE_PATTERN.match(foldername)
    if not m:
        raise ValueError(f"Bad sample folder: {foldername}")
    return int(m.group(1))

def build_logits_model(model, prefix):
    last = model.layers[-1]
    if isinstance(last, tf.keras.layers.Activation):
        return Model(model.input, last.input, name=prefix + "_logits")
    else:
        W, b = last.get_weights()
        pen = last.input
        logit_layer = layers.Dense(last.units, activation=None, name=prefix + "_logits_dense")
        out = logit_layer(pen)
        logit_layer.set_weights([W, b])
        return Model(model.input, out, name=prefix + "_logits")

# ───────────────────────────────────────────────────────────────────────────────
# 4. LOAD MODELS
# ───────────────────────────────────────────────────────────────────────────────

print("Loading MD model …")
md_model = tf.keras.models.load_model(MD_MODEL_PATH, custom_objects={"K": K})
print("Loading video model …")
vid_model = tf.keras.models.load_model(VID_MODEL_PATH)

md_logits_model  = build_logits_model(md_model, "md")
vid_logits_model = build_logits_model(vid_model, "vid")

# ───────────────────────────────────────────────────────────────────────────────
# 5. INFERENCE
# ───────────────────────────────────────────────────────────────────────────────

folders = sorted([d for d in os.listdir(BASE_VAL_DIR)
                  if SAMPLE_PATTERN.match(d)],
                 key=lambda x: int(SAMPLE_PATTERN.match(x).group(1)))
n = len(folders)
ids = np.zeros(n, dtype="int32")
md_probs   = np.zeros((n, NUM_CLASSES), dtype="float32")
md_logits  = np.zeros((n, NUM_CLASSES), dtype="float32")
vid_probs  = np.zeros((n, NUM_CLASSES), dtype="float32")
vid_logits = np.zeros((n, NUM_CLASSES), dtype="float32")

for i, f in enumerate(folders):
    sample_id = extract_id(f)
    ids[i] = sample_id
    d = os.path.join(BASE_VAL_DIR, f)

    # MD branch
    imgs = load_md_images_from_folder(d)
    preds_p = md_model.predict([img[np.newaxis,...] for img in imgs], verbose=0)[0]
    preds_z = md_logits_model.predict([img[np.newaxis,...] for img in imgs], verbose=0)[0]
    md_probs[i]  = preds_p
    md_logits[i] = preds_z

    # Video branch
    vol = load_video_h5_from_folder(d)
    vp = vid_model.predict(vol, verbose=0)[0]
    vz = vid_logits_model.predict(vol, verbose=0)[0]
    vid_probs[i]  = vp
    vid_logits[i] = vz

    if (i+1) % 500 == 0 or i+1 == n:
        print(f"Processed {i+1}/{n} samples")

print("Inference done.\n")

# ───────────────────────────────────────────────────────────────────────────────
# 6. SAVE PROB & LOGIT MATRICES
# ───────────────────────────────────────────────────────────────────────────────

def save_df(matrix, cols, name, to_excel=True):
    df = pd.DataFrame(matrix, columns=cols)
    df.insert(0, "id", ids)
    df = df.sort_values("id").reset_index(drop=True)
    csv_path = os.path.join(PROB_MATRIX_DIR, name + (".xlsx" if to_excel else ".csv"))
    if to_excel:
        df.to_excel(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

cols = [f"class_{i}" for i in range(NUM_CLASSES)]
# probabilities as Excel
save_df(md_probs,   cols, "prob_md_val",   to_excel=True)
save_df(vid_probs,  cols, "prob_video_val", to_excel=True)
# logits as Excel
save_df(md_logits,  cols, "logit_md_val",   to_excel=True)
save_df(vid_logits, cols, "logit_video_val",to_excel=True)

print("\nAll probability files saved to:", PROB_MATRIX_DIR)

# ───────────────────────────────────────────────────────────────────────────────
# 7. SAVE PREDICTIONS CSVs
# ───────────────────────────────────────────────────────────────────────────────

# MD-only
md_pred = np.argmax(md_probs, axis=1)
md_df = pd.DataFrame({"id": ids, "pred": md_pred}).sort_values("id").reset_index(drop=True)
md_df.to_csv(os.path.join(PRED_DIR, "md_preds_val.csv"), index=False)
print("Saved md_preds.csv for validation data")

# Video-only
vid_pred = np.argmax(vid_probs, axis=1)
vd_df = pd.DataFrame({"id": ids, "pred": vid_pred}).sort_values("id").reset_index(drop=True)
vd_df.to_csv(os.path.join(PRED_DIR, "video_preds_val.csv"), index=False)
print("Saved video_preds.csv for validation data")

# Fusion (0.5/0.5)
fuse = 0.5 * md_probs + 0.5 * vid_probs
fu_pred = np.argmax(fuse, axis=1)
fu_df = pd.DataFrame({"id": ids, "pred": fu_pred}).sort_values("id").reset_index(drop=True)
fu_df.to_csv(os.path.join(PRED_DIR, "fusion_preds_val.csv"), index=False)
print("Saved fusion_preds.csv for validation data")

print("\nAll validation prediction files saved to:", PRED_DIR)
