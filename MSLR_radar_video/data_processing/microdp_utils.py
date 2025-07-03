import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_microdoppler(
    video_path: str,
    interp_factor: int,
    roi_start: int,
    roi_end: int,
    freq_start: int,
    freq_end: int,
    out_size: tuple
) -> np.ndarray:
    """
    Reads an RDM video, computes the micro-Doppler spectrogram,
    normalizes, and resizes to out_size.

    Returns:
        spec_resized: 2D array of shape out_size with values in [0,1]
        or None if no frames could be read.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames.append(gray)
    cap.release()
    if not frames:
        return None

    # Stack frames into (T, H, W)
    vid3 = np.stack(frames, axis=0)
    T, H, W = vid3.shape
    print("Height (H):", H)
    print("Width  (W):", W)

    # Temporal interpolation
    Tn = T * interp_factor
    interp = np.zeros((Tn, H, W), dtype=np.float32)
    for t_new in range(Tn):
        t = t_new / interp_factor
        t0 = int(np.floor(t))
        t1 = min(t0 + 1, T - 1)
        alpha = t - t0
        interp[t_new] = (1 - alpha) * vid3[t0] + alpha * vid3[t1]

    # Crop range rows and collapse to micro-Doppler
    roi = interp[:, roi_start:roi_end, :]
    micro = np.sum(roi, axis=1)  # shape (Tn, W)
    micro = micro[:, freq_start:freq_end]

    # Normalize to [0, 1]
    micro -= micro.min()
    micro /= (micro.max() + 1e-6)

    # Transpose to (freq_bins, time_frames) and resize
    spec = micro.T  # shape (freq_bins, time_frames)
    spec_resized = cv2.resize(
        spec,
        out_size,
        interpolation=cv2.INTER_AREA
    )
    return spec_resized


def save_spectrogram(
    spec: np.ndarray,
    out_png: str,
    vmin: float,
    vmax: float
):
    """
    Saves a 2D spectrogram array as a JET-colormapped PNG with fixed vmin/vmax.

    Args:
        spec: 2D array of shape (H, W) with values in [0,1]
        out_png: output file path
        vmin, vmax: color scale limits
    """
    fig, ax = plt.subplots(
        figsize=(spec.shape[1] / 100, spec.shape[0] / 100),
        dpi=100
    )
    ax.imshow(
        spec,
        origin='lower',
        aspect='auto',
        cmap='jet',
        vmin=vmin,
        vmax=vmax
    )
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_png, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
