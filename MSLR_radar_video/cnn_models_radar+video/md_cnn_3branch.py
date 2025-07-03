import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K, callbacks
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Flatten, Dense, Dropout,
    BatchNormalization, ReLU, Lambda, Concatenate
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────

BASE_DIR     = r"C:\Users\sb3682.ECE-2V7QHQ3\My Stuff\kaggle_comp\Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Train Val\train"
NUM_CLASSES  = 126
VALID_SPLIT  = 0.2
RANDOM_SEED  = 42

IMG_SIZE     = (96, 96)            # MD image dimensions
INPUT_SHAPE  = (*IMG_SIZE, 1)        # single-channel grayscale

BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-3

# Directory to save checkpoints every 5 epochs
CHECKPOINT_DIR = "checkpoints_md_cnn"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# 1. COLLECT & SPLIT SAMPLES
# ───────────────────────────────────────────────────────────────────────────────

def collect_all_samples(base_dir: str, num_classes: int):
    samples = []
    for cls_folder in sorted(os.listdir(base_dir)):
        cls_path = os.path.join(base_dir, cls_folder)
        if not os.path.isdir(cls_path):
            continue
        parts = cls_folder.split("_")
        try:
            class_idx = int(parts[0])
        except:
            continue
        if class_idx >= num_classes:
            continue

        for sample_folder in sorted(os.listdir(cls_path)):
            if not sample_folder.startswith("SAMPLE_"):
                continue
            sample_path = os.path.join(cls_path, sample_folder)
            if not os.path.isdir(sample_path):
                continue
            png_files = [f for f in os.listdir(sample_path) if f.lower().endswith(".png")]
            if len(png_files) < 3:
                continue
            sample_num = sample_folder.split("_")[1]
            id_string = f"{class_idx}_{sample_num}"
            samples.append((class_idx, sample_path, id_string))
    return samples

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

all_samples = collect_all_samples(BASE_DIR, NUM_CLASSES)
labels = [s[0] for s in all_samples]
train_samples, val_samples = train_test_split(
    all_samples,
    test_size=VALID_SPLIT,
    stratify=labels,
    random_state=RANDOM_SEED
)

# ───────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING: LOAD GRAYSCALE MD IMAGES (128×128)
# ───────────────────────────────────────────────────────────────────────────────

def preprocess_md_grayscale(sample_path: str):
    png_files = sorted([f for f in os.listdir(sample_path) if f.lower().endswith(".png")])[:3]
    imgs = []
    for fname in png_files:
        img = cv2.imread(os.path.join(sample_path, fname), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = img.astype("float32") / 255.0
        imgs.append(img[..., np.newaxis])  # (128,128,1)
    while len(imgs) < 3:
        imgs.append(np.zeros((*IMG_SIZE, 1), dtype="float32"))
    return imgs[0], imgs[1], imgs[2]

def build_md_arrays(sample_list):
    x0_list, x1_list, x2_list, y_list, ids = [], [], [], [], []
    for class_idx, sample_path, id_str in sample_list:
        md0, md1, md2 = preprocess_md_grayscale(sample_path)
        x0_list.append(md0)
        x1_list.append(md1)
        x2_list.append(md2)
        onehot = np.zeros(NUM_CLASSES, dtype="float32")
        onehot[class_idx] = 1.0
        y_list.append(onehot)
        ids.append(id_str)
    x0_arr = np.stack(x0_list, axis=0)
    x1_arr = np.stack(x1_list, axis=0)
    x2_arr = np.stack(x2_list, axis=0)
    y_arr  = np.stack(y_list, axis=0)
    return x0_arr, x1_arr, x2_arr, y_arr, ids

x0_tr, x1_tr, x2_tr, y_tr, ids_tr = build_md_arrays(train_samples)
x0_val, x1_val, x2_val, y_val, ids_val = build_md_arrays(val_samples)

# ───────────────────────────────────────────────────────────────────────────────
# 3. ENHANCED MD‐CNN (three‐branch relational CNN)
# ───────────────────────────────────────────────────────────────────────────────

def make_branch_enhanced(name):
    inp = Input(shape=INPUT_SHAPE, name=f"inp_{name}")
    x   = Conv2D(64, (3,3), padding='same', use_bias=False)(inp)
    x   = BatchNormalization()(x)
    x   = ReLU()(x)
    x   = MaxPool2D((2,2))(x)   # → 64×64

    x   = Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x   = BatchNormalization()(x)
    x   = ReLU()(x)
    x   = MaxPool2D((2,2))(x)   # → 32×32

    x   = Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x   = BatchNormalization()(x)
    x   = ReLU()(x)
    x   = MaxPool2D((2,2))(x)   # → 16×16

    x   = Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x   = BatchNormalization()(x)
    x   = ReLU()(x)
    x   = MaxPool2D((2,2))(x)   # → 8×8

    x   = Flatten()(x)          # → 8*8*128 = 8192
    return inp, x

def build_md_cnn_enhanced(num_classes=NUM_CLASSES, lr=LR):
    inp0, f0 = make_branch_enhanced("ant0")
    inp1, f1 = make_branch_enhanced("ant1")
    inp2, f2 = make_branch_enhanced("ant2")

    diff01 = Lambda(lambda t: K.abs(t[0] - t[1]), name="diff01")([f0, f1])
    diff02 = Lambda(lambda t: K.abs(t[0] - t[1]), name="diff02")([f0, f2])
    diff12 = Lambda(lambda t: K.abs(t[0] - t[1]), name="diff12")([f1, f2])

    prod01 = Lambda(lambda t: t[0] * t[1], name="prod01")([f0, f1])
    prod02 = Lambda(lambda t: t[0] * t[1], name="prod02")([f0, f2])
    prod12 = Lambda(lambda t: t[0] * t[1], name="prod12")([f1, f2])

    merged = Concatenate(name="fusion")([
        f0, f1, f2,
        diff01, diff02, diff12,
        prod01, prod02, prod12
    ])

    x = Dense(512, use_bias=False)(merged)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    out = Dense(num_classes, activation='softmax', name="md_output")(x)

    model = models.Model(inputs=[inp0, inp1, inp2], outputs=out, name="RelationalMD_CNN_Enhanced")
    model.compile(
        optimizer=Adam(lr, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

md_model = build_md_cnn_enhanced()
md_model.summary()

# ───────────────────────────────────────────────────────────────────────────────
# 4. CALLBACK: Save model every 5 epochs with validation accuracy in filename
# ───────────────────────────────────────────────────────────────────────────────

class SaveEveryFive(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs['val_accuracy'] contains the validation accuracy of this epoch
        if (epoch + 1) % 5 == 0:
            val_acc = logs.get('val_accuracy', 0.0)
            filename = f"epoch_{epoch+1:03d}_valAcc_{val_acc:.4f}.h5"
            filepath = os.path.join(CHECKPOINT_DIR, filename)
            self.model.save(filepath)
            print(f"\n  → Saved checkpoint: {filepath}")

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6
)
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=False, verbose=1
)
save_every_five = SaveEveryFive()

# ───────────────────────────────────────────────────────────────────────────────
# 5. TRAINING
# ───────────────────────────────────────────────────────────────────────────────

history = md_model.fit(
    [x0_tr, x1_tr, x2_tr], y_tr,
    validation_data=([x0_val, x1_val, x2_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[reduce_lr, early_stop, save_every_five]
)

# ───────────────────────────────────────────────────────────────────────────────
# 6. FINAL EVALUATION & MODEL SAVE
# ───────────────────────────────────────────────────────────────────────────────

val_loss, val_acc = md_model.evaluate([x0_val, x1_val, x2_val], y_val, batch_size=BATCH_SIZE, verbose=2)
print(f"\nFinal Enhanced MD-CNN Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

final_model_filename = f"relational_md_cnn_enhanced_final_valAcc_{val_acc:.4f}.h5"
md_model.save(final_model_filename)
print(f"Final model saved as: {final_model_filename}")

# Save validation predictions (optional)
y_pred = md_model.predict([x0_val, x1_val, x2_val], batch_size=BATCH_SIZE, verbose=0)
pred_labels = np.argmax(y_pred, axis=1)
df_preds = pd.DataFrame({
    'file_id': ids_val,
    'prediction': pred_labels
})
df_preds.to_excel("md_val_predictions_enhanced.xlsx", index=False)
print("Validation predictions saved to: md_val_predictions_enhanced.xlsx")
