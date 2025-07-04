# ICCV2025_MSLR

This repository contains code for **Italian Sign Language Recognition (MSLR)** prepared for the **ICCV 2025 Challenge**.  
It integrates predictions from **three complementary modalities**—**skeleton**, **radar**, and **RGB video**—to improve isolated sign classification performance.

Our Proposed Multi-modal Framework includes:
- **Skeleton Dynamics**
- **Radar Micro-Doppler Signature**
- **RGB Video Sequenes**


---

## 📁 Project Structure

- `Pose_estimation/` – Scripts for extracting 2D pose/keypoints using HRNet  
- `SL-GCN/` – Skeleton-based Graph Convolutional Network (from SAM-SLR)  
- `Fusion/` – Logits generated from SL-GCN for fusion  
- `MSLR_radar_video/` – Radar and RGB video CNN classifiers and scripts  
- `Final_prediction/` – Final CSV after weighted fusion of all three modalities  
- `Multimodal_fusion.py` – Script to perform fusion and save final predictions  

---

## 🔀 Multimodal Fusion Results

| **Multi-modal Fusion**        | **Val Top-1** | **Test Top-1** |
|-------------------------------|---------------|----------------|
| Radar + Video                 | 97.19%        | 97.42%         |
| Radar + Skeletal              | 99.37%        | 99.44%         |
| Skeletal + Video              | 99.50%        | 99.55%         |
| **Skeletal + Video + Radar**  | **99.61%**    | **99.71%**     |

> *Table: Top-1 accuracy (%) on validation and test set.*

---

## 🛠️ Usage 

Instructions to reproduce our results and run inference. The general pipeline is:

1. **Pose_estimation**  
   Extract 2D human pose keypoints from RGB video frames using HRNet.

2. **SL-GCN**  
   Use the extracted skeletons as input to the SL-GCN model for gesture classification.

3. **MSLR_radar_video**  
   Use radar and video data with CNN models to produce predictions.

4. **Fusion**  
   Run the fusion script to combine the three modalities:
   ```bash
   python Multimodal_fusion.py




## 📜 License
Parts of this repo (e.g., SL-GCN) is reused from [SAM-SLR](https://github.com/jackyjsy/CVPR21Chal-SLR), licensed under **Creative Commons Zero v1.0 Universal**, with the following added restriction:

> **This code is released for academic research use only. Commercial use is prohibited.**

Please cite the original SAM-SLR paper when using SL-GCN.

## 📚 Citation

```bibtex
@inproceedings{jiang2021skeleton,
  title={Skeleton Aware Multi-modal Sign Language Recognition},
  author={Jiang, Songyao and Sun, Bin and Wang, Lichen and Bai, Yue and Li, Kunpeng and Fu, Yun},
  booktitle={CVPR Workshops},
  year={2021}
}

@article{jiang2021sign,
  title={Sign Language Recognition via Skeleton-Aware Multi-Model Ensemble},
  author={Jiang, Songyao and Sun, Bin and Wang, Lichen and Bai, Yue and Li, Kunpeng and Fu, Yun},
  journal={arXiv preprint arXiv:2110.06161},
  year={2021}
}
