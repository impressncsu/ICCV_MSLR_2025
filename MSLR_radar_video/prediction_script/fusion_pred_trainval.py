"""
fusion_manual_with_logits.py

Performs α‐weighted fusion between two pretrained Keras models (ResNet(2+1)D and 3‐branch MD‐CNN)
over a validation set, and also extracts the raw logits (pre‐softmax scores) from both models.
Saves four 2D matrices (probs_res, probs_md, logits_res, logits_md) into the specified folder.

Usage:
 1. Install dependencies if needed:
      pip install tensorflow pandas openpyxl numpy opencv-python h5py
 2. Place this script anywhere (no need to import your custom build functions).
 3. Update the four path variables below:
      • BASE_DIR
      • VAL_EXCEL_PATH
      • PRETRAINED_RESNET
      • PRETRAINED_MD
 4. Run:
      python fusion_manual_with_logits.py
"""

import os
import h5py
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# ───────────────────────────────────────────────────────────────────────────────
# 0. IMPORT KERAS BACKEND AS K (so any Lambda layers referencing "K" will work)
# ───────────────────────────────────────────────────────────────────────────────
from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers

# ───────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION: update these paths to your setup
# ───────────────────────────────────────────────────────────────────────────────

################ modify the directories accordingly ###########################

BASE_DIR           = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Train Val/train"
VAL_EXCEL_PATH     = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/prediction_script/validation_samples.xlsx"
PRETRAINED_RESNET  = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/final_weights/video_final.h5"
PRETRAINED_MD      = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/final_weights/radar_final.h5"

PROB_MATRIX_DIR    = r"C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/results"
os.makedirs(PROB_MATRIX_DIR, exist_ok=True)

OUT_CSV            = "alpha_vs_accuracy.csv"

FRAME_TARGET       = 64    # number of frames in each .h5 volume
SPATIAL_SIZE       = 96    # width/height of each frame
IMG_SIZE           = 96    # MD‐CNN expects 96×96 grayscale inputs
NUM_CLASSES        = 126   # number of output classes

# ───────────────────────────────────────────────────────────────────────────────
# 2. LOAD the two pretrained models
# ───────────────────────────────────────────────────────────────────────────────
print("Loading pretrained ResNet(2+1)D from:", PRETRAINED_RESNET)
res_model = tf.keras.models.load_model(PRETRAINED_RESNET, compile=False)
print("Loading pretrained MD‐CNN (3-branch) from:", PRETRAINED_MD)
md_model  = tf.keras.models.load_model(PRETRAINED_MD, compile=False)

# sanity‐check
assert res_model.output_shape[-1] == NUM_CLASSES, "ResNet: wrong output size"
assert md_model.output_shape[-1]  == NUM_CLASSES, "MD‐CNN: wrong output size"

# ───────────────────────────────────────────────────────────────────────────────
# 2.5 BUILD logits‐only models for pre‐softmax extraction
# ───────────────────────────────────────────────────────────────────────────────
def build_logits_model(model, name_prefix):
    last = model.layers[-1]
    if isinstance(last, tf.keras.layers.Activation):
        # separate Activation softmax
        return Model(inputs=model.input, outputs=last.input, name=f"{name_prefix}_logits")
    else:
        # fused Dense(..., activation='softmax')
        W, b = last.get_weights()
        penult = last.input
        logits_layer = layers.Dense(last.units, activation=None,
                                    name=f"{name_prefix}_logits_dense")
        logits_out   = logits_layer(penult)
        logits_layer.set_weights([W, b])
        return Model(inputs=model.input, outputs=logits_out, name=f"{name_prefix}_logits")

res_logits_model = build_logits_model(res_model, "res")
md_logits_model  = build_logits_model(md_model,  "md")

# ───────────────────────────────────────────────────────────────────────────────
# 3. READ validation_samples.xlsx
# ───────────────────────────────────────────────────────────────────────────────
df_val = pd.read_excel(VAL_EXCEL_PATH)
if not {"folder_name", "sample_number"}.issubset(df_val.columns):
    raise ValueError("validation_samples.xlsx must have columns: folder_name, sample_number")

df_val["gt_label"] = df_val["folder_name"].str.split("_", 1).str[0].astype(int)
N_val = len(df_val)
print(f"\nLoaded {N_val} validation entries.\n")

# ───────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE LOOP: collect probs & logits
# ───────────────────────────────────────────────────────────────────────────────
probs_res_list   = []
probs_md_list    = []
logits_res_list  = []
logits_md_list   = []
gt_list          = []

for idx, row in df_val.iterrows():
    folder = row["folder_name"]
    sample = row["sample_number"]
    gt     = row["gt_label"]
    sample_dir = os.path.join(BASE_DIR, folder, sample)
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(sample_dir)

    # --- ResNet(2+1)D volume inference ---
    h5_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(".h5")]
    if len(h5_files) != 1:
        raise FileNotFoundError(f"{sample_dir} → expected 1 .h5, got {h5_files}")
    with h5py.File(os.path.join(sample_dir, h5_files[0]), "r") as f:
        volume = f["video"][()].astype(np.float32)
    # normalize & pad/crop
    v_min, v_max = volume.min(), volume.max()
    volume = (volume - v_min) / max(1e-8, (v_max - v_min))
    if volume.shape[0] < FRAME_TARGET:
        volume = np.pad(volume, ((0, FRAME_TARGET - volume.shape[0]), (0,0), (0,0)),
                        constant_values=0.0)
    else:
        volume = volume[:FRAME_TARGET]
    h, w = volume.shape[1], volume.shape[2]
    pad_h, pad_w = max(0, SPATIAL_SIZE - h), max(0, SPATIAL_SIZE - w)
    volume = np.pad(volume, ((0,0),(0,pad_h),(0,pad_w)), constant_values=0.0)[:,
                :SPATIAL_SIZE, :SPATIAL_SIZE]
    volume = volume[..., np.newaxis][np.newaxis, ...]

    prob_res  = res_model.predict(volume, verbose=0)[0]
    logit_res = res_logits_model.predict(volume, verbose=0)[0]

    # --- MD‐CNN (3‐branch) inference ---
    pngs = sorted(f for f in os.listdir(sample_dir) if f.lower().endswith(".png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG in {sample_dir}")
    md_inputs = []
    for i in range(3):
        if i < len(pngs):
            img = cv2.imread(os.path.join(sample_dir, pngs[i]), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            md_inputs.append(img[..., np.newaxis][np.newaxis, ...])
        else:
            md_inputs.append(np.zeros((1,IMG_SIZE,IMG_SIZE,1), dtype=np.float32))

    prob_md  = md_model.predict(md_inputs, verbose=0)[0]
    logit_md = md_logits_model.predict(md_inputs, verbose=0)[0]

    probs_res_list.append(prob_res.astype(np.float32))
    logits_res_list.append(logit_res.astype(np.float32))
    probs_md_list.append(prob_md.astype(np.float32))
    logits_md_list.append(logit_md.astype(np.float32))
    gt_list.append(gt)

print("Inference loop complete.\n")

# ───────────────────────────────────────────────────────────────────────────────
# 5. STACK into arrays
# ───────────────────────────────────────────────────────────────────────────────
probs_res_arr  = np.stack(probs_res_list,  axis=0)
logits_res_arr = np.stack(logits_res_list, axis=0)
probs_md_arr   = np.stack(probs_md_list,   axis=0)
logits_md_arr  = np.stack(logits_md_list,  axis=0)
gt_arr         = np.array(gt_list, dtype=np.int32)

# sample names vector
sample_names = df_val.apply(lambda r: f"{r['folder_name']}/{r['sample_number']}", axis=1)

# ───────────────────────────────────────────────────────────────────────────────
# 6. SAVE 4 MATRICES TO CSV
# ───────────────────────────────────────────────────────────────────────────────
def save_matrix(arr, filename):
    df = pd.DataFrame(arr, columns=[f"class_{i}" for i in range(NUM_CLASSES)])
    df.insert(0, "sample", sample_names)
    out_path = os.path.join(PROB_MATRIX_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

save_matrix(probs_res_arr,  "probs_video_train.csv")
save_matrix(logits_res_arr, "logits_video_train.csv")
save_matrix(probs_md_arr,   "probs_md_train.csv")
save_matrix(logits_md_arr,  "logits_md_train.csv")

# ───────────────────────────────────────────────────────────────────────────────
# 7. Standalone accuracies
# ───────────────────────────────────────────────────────────────────────────────
acc_res = np.mean(np.argmax(probs_res_arr, axis=1) == gt_arr)
acc_md  = np.mean(np.argmax(probs_md_arr,  axis=1) == gt_arr)
print(f"ResNet accuracy: {acc_res*100:.2f}%")
print(f"MD‐CNN accuracy: {acc_md*100:.2f}%\n")

# ───────────────────────────────────────────────────────────────────────────────
# 8. α‐weighted fusion
# ───────────────────────────────────────────────────────────────────────────────
alphas, results = np.arange(0,1.0001,0.1), []
print(" α    |  Fused Acc")
print("-------------------")
for α in alphas:
    fused = α*probs_res_arr + (1-α)*probs_md_arr
    acc   = np.mean(np.argmax(fused,axis=1)==gt_arr)
    results.append({"alpha": float(α), "accuracy": float(acc)})
    print(f"{α:0.2f} | {acc*100:5.2f}%")
print("-------------------\n")

# ───────────────────────────────────────────────────────────────────────────────
# 9. Save α vs. accuracy
# ───────────────────────────────────────────────────────────────────────────────
pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"Saved α vs. accuracy → '{OUT_CSV}'")
