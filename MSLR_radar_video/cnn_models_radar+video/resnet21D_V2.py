# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 15:39:25 2025

@author: sb3682
"""

import os
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ───────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & SEEDS
# ───────────────────────────────────────────────────────────────────────────────
BASE_TRAIN_DIR = r"C:\Users\sb3682.ECE-2V7QHQ3\My Stuff\kaggle_comp\Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Train Val\train"

RANDOM_SEED    = 42
NUM_CLASSES    = 126
BATCH_SIZE     = 32
NUM_EPOCHS     = 200
VALID_SPLIT    = 0.05

# Updated target dimensions:
FRAME_TARGET   = 64    # now 32 frames
SPATIAL_SIZE   = 96   # now 128×128 spatial

LEARNING_RATE  = 1e-3

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Directory to save checkpoints
CHECKPOINT_DIR = "checkpoints_r2p1d"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# 2. COLLECT & SPLIT SAMPLES
# ───────────────────────────────────────────────────────────────────────────────
def collect_all_samples(base_dir: str, num_classes: int):
    samples = []
    for class_folder in sorted(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        parts = class_folder.split("_")
        try:
            class_idx = int(parts[0])
        except ValueError:
            continue
        if class_idx >= num_classes:
            continue

        for sample_folder in sorted(os.listdir(class_path)):
            if not sample_folder.startswith("SAMPLE_"):
                continue
            samp_path = os.path.join(class_path, sample_folder)
            if not os.path.isdir(samp_path):
                continue
            # Expect exactly one .h5 per sample
            h5_files = [f for f in os.listdir(samp_path) if f.lower().endswith(".h5")]
            if not h5_files:
                continue
            sample_num = sample_folder.split("_")[1]
            id_str = f"{class_idx}_{sample_num}"
            samples.append((class_idx, samp_path, id_str))
    return samples

all_samples = collect_all_samples(BASE_TRAIN_DIR, NUM_CLASSES)
labels = [s[0] for s in all_samples]
train_samples, val_samples = train_test_split(
    all_samples,
    test_size=VALID_SPLIT,
    stratify=labels,
    random_state=RANDOM_SEED
)
print("Train samples:", len(train_samples), "Validation samples:", len(val_samples))

# ───────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADING (Normalize to [0,1], resize if needed)
# ───────────────────────────────────────────────────────────────────────────────
def load_h5_video(sample_tuple):
    cls, path, _ = sample_tuple
    h5_fname = next(f for f in os.listdir(path) if f.lower().endswith(".h5"))
    with h5py.File(os.path.join(path, h5_fname), "r") as f:
        vol = f["video"][()].astype(np.float32)  # shape: (orig_frames, orig_H, orig_W)
    # Normalize to [0,1]
    vol -= np.min(vol)
    vol /= (np.max(vol) + 1e-8)

    # If volumetric shape differs, crop/pad or resize:
    # For simplicity, assume H5 already is (FRAME_TARGET, SPATIAL_SIZE, SPATIAL_SIZE).
    # Otherwise, crop/resize as needed:
    if vol.shape != (FRAME_TARGET, SPATIAL_SIZE, SPATIAL_SIZE):
        # Crop or pad depth
        vol = vol[:FRAME_TARGET, :SPATIAL_SIZE, :SPATIAL_SIZE]
        # If smaller than required, pad with zeros
        pad_d = FRAME_TARGET - vol.shape[0]
        pad_h = SPATIAL_SIZE - vol.shape[1]
        pad_w = SPATIAL_SIZE - vol.shape[2]
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            vol = np.pad(
                vol,
                ((0, pad_d if pad_d>0 else 0),
                 (0, pad_h if pad_h>0 else 0),
                 (0, pad_w if pad_w>0 else 0)),
                mode="constant", constant_values=0
            )
    return vol[..., np.newaxis], cls  # final shape: (32,128,128,1)

def gen_train():
    for entry in train_samples:
        v, lbl = load_h5_video(entry)
        yield v, lbl

def gen_val():
    for entry in val_samples:
        v, lbl = load_h5_video(entry)
        yield v, lbl

output_types  = (tf.float32, tf.int32)
output_shapes = (
    tf.TensorShape([FRAME_TARGET, SPATIAL_SIZE, SPATIAL_SIZE, 1]),
    tf.TensorShape([])
)

train_ds = (
    tf.data.Dataset.from_generator(gen_train, output_types=output_types, output_shapes=output_shapes)
    .shuffle(buffer_size=max(len(train_samples), 100), seed=RANDOM_SEED)
    .repeat()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_generator(gen_val, output_types=output_types, output_shapes=output_shapes)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

steps_per_epoch = len(train_samples) // BATCH_SIZE

# ───────────────────────────────────────────────────────────────────────────────
# 4. ResNet(2+1)D WITH BatchNormalization (adapted for 32×128×128)
# ───────────────────────────────────────────────────────────────────────────────
def conv2plus1d_block(x, filters, stride=1, downsample=False):
    shortcut = x

    # Spatial: 1×3×3
    x = layers.Conv3D(filters, (1, 3, 3),
                      strides=(1, stride, stride),
                      padding="same", use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Temporal: 3×1×1
    x = layers.Conv3D(filters, (3, 1, 1),
                      strides=(stride, 1, 1),
                      padding="same", use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    # No activation yet

    # Downsample shortcut if needed
    if downsample:
        shortcut = layers.Conv3D(filters, (1, 1, 1),
                                 strides=(stride, stride, stride),
                                 padding="same", use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-5))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet2p1d(input_shape=(FRAME_TARGET, SPATIAL_SIZE, SPATIAL_SIZE, 1)):
    inp = layers.Input(shape=input_shape)

    # Initial conv: 16 filters, (3×7×7), downsample spatial dims
    x = layers.Conv3D(16, (3, 7, 7),
                      strides=(1, 2, 2),
                      padding="same", use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D((1, 3, 3),
                        strides=(1, 2, 2),
                        padding="same")(x)

    # Stage 1: filters=16, two blocks (no downsample)
    x = conv2plus1d_block(x, filters=16, stride=1, downsample=False)
    x = conv2plus1d_block(x, filters=16, stride=1, downsample=False)

    # Stage 2: filters=32, first block downsamples spatially
    x = conv2plus1d_block(x, filters=32, stride=2, downsample=True)
    x = conv2plus1d_block(x, filters=32, stride=1, downsample=False)

    # Stage 3: filters=64, first block downsamples spatially
    x = conv2plus1d_block(x, filters=64, stride=2, downsample=True)
    x = conv2plus1d_block(x, filters=64, stride=1, downsample=False)

    # Global pooling + head
    x = layers.GlobalAveragePooling3D()(x)   # → (64,)
    x = layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs=inp, outputs=out, name="ResNet2p1D_BN_32x128")

model = build_resnet2p1d()
model.summary()

optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ───────────────────────────────────────────────────────────────────────────────
# 5. CALLBACKS: Save every 5 epochs, ReduceLROnPlateau, EarlyStopping
# ───────────────────────────────────────────────────────────────────────────────
class SaveEveryFive(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            val_acc = logs.get('val_accuracy', 0.0)
            fname = f"epoch_{epoch+1:03d}_valAcc_{val_acc:.4f}.h5"
            path = os.path.join(CHECKPOINT_DIR, fname)
            self.model.save(path)
            print(f"\n  → Checkpoint saved: {path}")

save_every_five = SaveEveryFive()

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=10, verbose=1, min_lr=1e-6
)
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=30, restore_best_weights=True, verbose=1
)

# ───────────────────────────────────────────────────────────────────────────────
# 6. TRAINING
# ───────────────────────────────────────────────────────────────────────────────
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=[save_every_five, reduce_lr, early_stop]
)

# ───────────────────────────────────────────────────────────────────────────────
# 7. FINAL VALIDATION PREDICTIONS → EXCEL
# ───────────────────────────────────────────────────────────────────────────────
val_ids, val_preds = [], []
for cls_idx, sample_path, id_str in val_samples:
    vol, _ = load_h5_video((cls_idx, sample_path, id_str))
    vol    = np.expand_dims(vol, axis=0)          # (1,32,128,128,1)
    pred   = model.predict(vol, verbose=0)[0]     # (NUM_CLASSES,)
    val_ids.append(id_str)
    val_preds.append(int(np.argmax(pred)))

pd.DataFrame({"id": val_ids, "prediction": val_preds}) \
  .to_excel("r2p1d_final_preds.xlsx", index=False)
print("Saved r2p1d_final_preds.xlsx")
