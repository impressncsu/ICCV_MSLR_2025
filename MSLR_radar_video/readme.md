# Multimodal Italian Sign Language Recognition

This repository contains scripts and models for a multimodal sign language recognition system, combining radar-based micro-Doppler signatures, video sequences, and skeletal data, based on our ICCV 2025 paper titled "A Multimodal Video and Radar Fusion Framework for High-Accuracy Isolated Sign Language Recognition."

---

## Folder Structure

### 1. **cnn\_model\_radar+video**

This folder contains:

* CNN models for radar-based and video-based architectures.
* During training, a checkpoint folder will be created to store model weights after every 5 epochs.

### 2. **final\_weights**

Contains the final trained weights for both radar and video models.

Download the final model weights from the following location:

[Final Model Weights (Google Drive)](https://drive.google.com/drive/folders/1oSWKC3bShKOiI5yynSaQuwl1nnKQfRKf)

### 3. **prediction\_scripts**

Includes scripts for generating predictions and fusion:

* **fusion\_pred\_train\_val.py**

  * Generates probability matrices from trained radar and video models.
  * Produces four CSV files in the results folder: `logit_md_train.csv`, `logit_video_train.csv`, `probs_md_train.csv`, and `probs_video_train.csv`.
  * Performs decision-level fusion, calculates optimal fusion weights, and saves them as a CSV file.

* **test\_fusion\_pred.py**

  * Evaluates test samples using model weights.
  * Saves radar, video, and fused predictions in the `results/predictions` folder.

* **val\_fusion\_pred.py**

  * Evaluates validation samples similarly to the test script.

* **validation\_samples.xlsx**

  * Lists the 20% validation samples separated during training.

### 4. **fusion\_with\_slgcnn**

This directory handles fusion with skeletal data from the SL-GCN model:

* Processes logit matrices from radar, video, and SL-GCN models.
* Performs fusion combinations:

  * SL-GCN + Radar
  * SL-GCN + Video
  * SL-GCN + Video + Radar
* Saves validation and test predictions within a newly created `predictions` folder.

### 5. **data\_processing**

This folder contains scripts for preprocessing data:

* **train\_md\_gen.py** and **val\_md\_gen.py**

  * These scripts process the training and validation/test data respectively for radar Doppler maps (RDMs).
  * Use the function in **microdp\_utils.py** to generate micro-Doppler images from the RDMs.
  * Generated PNG files are saved in the same location as the original RDM files.

* **train\_video\_resized.py** and **val\_video\_resized.py**

  * Resize the video data.
  * Save resized videos as `.h5` files in the same directories as the original `.mkv` video files.

---

## Fusion Results

| Fusion Combination       | Validation Accuracy (%) | Test Accuracy (%) |
| ------------------------ | ----------------------- | ----------------- |
| Radar + Video            | 97.19                   | 97.42             |
| Radar + Skeletal         | 99.37                   | 99.44             |
| Video + Skeletal         | 99.50                   | 99.55             |
| Radar + Video + Skeletal | **99.61**               | **99.71**         |

The table above demonstrates the effectiveness of multimodal fusion, highlighting that the combination of radar, video, and skeletal data provides superior accuracy. The skeletal modality plays a crucial role in performance, while the radar and video modalities add complementary robustness, especially under challenging visual conditions.

---

## How to Use

### Data Processing

* Navigate to `data_processing` and run:

  * `train_md_gen.py` and `val_md_gen.py` to generate micro-Doppler images.
  * `train_video_resized.py` and `val_video_resized.py` to resize video files.

### Training

* Navigate to `cnn_model_radar+video` and run the respective training scripts for radar and video models.
* Monitor the training progress through checkpoints.

### Prediction and Fusion

1. Navigate to the `prediction_scripts` folder.
2. Run `fusion_pred_train_val.py` to obtain initial probability matrices and optimal fusion weights.
3. Use `test_fusion_pred.py` and `val_fusion_pred.py` for generating predictions on respective datasets.

### Final Fusion

* For comprehensive multimodal fusion, navigate to `fusion_with_slgcnn` and execute the fusion scripts.

---