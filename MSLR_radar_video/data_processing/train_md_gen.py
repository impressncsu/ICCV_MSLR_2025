# -*- coding: utf-8 -*-
"""
script.py

Generate 128×128 training micro-Doppler spectrograms with consistent
normalization and JET colormap.

Usage:
  python script.py
"""

import os
from microdp_utils import compute_microdoppler, save_spectrogram

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR      = r'C:\Users\sb3682.ECE-2V7QHQ3\My Stuff\kaggle_comp\Sessioni CHALLENGE ICCV MSLR 2025 Track 2 - Train Val\train'
INTERP_FACTOR = 2
ROI_START     = 50
ROI_END       = 200
FREQ_START    = 300
FREQ_END      = 724

OUT_SIZE = (128, 128)
VMIN, VMAX = 0.0, 1.0

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
for cls in sorted(os.listdir(BASE_DIR)):
    cls_dir = os.path.join(BASE_DIR, cls)
    if not os.path.isdir(cls_dir):
        continue

    for sample in sorted(os.listdir(cls_dir)):
        sample_dir = os.path.join(cls_dir, sample)
        if not os.path.isdir(sample_dir):
            continue

        for fname in sorted(os.listdir(sample_dir)):
            if fname.lower().endswith('.mp4') and 'RDM' in fname.upper():
                in_vid = os.path.join(sample_dir, fname)
                base   = os.path.splitext(fname)[0]
                out_png = os.path.join(sample_dir, f"{base}.png")

                spec = compute_microdoppler(
                    in_vid,
                    interp_factor=INTERP_FACTOR,
                    roi_start=ROI_START, roi_end=ROI_END,
                    freq_start=FREQ_START, freq_end=FREQ_END,
                    out_size=OUT_SIZE
                )
                if spec is None:
                    continue

                save_spectrogram(
                    spec, out_png,
                    vmin=VMIN, vmax=VMAX
                )

        print(f"Processed train sample folder: {sample_dir}")
