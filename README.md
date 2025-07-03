# ICCV2025_MSLR

This repository contains code for **Multimodal Sign Language Recognition (MSLR)** prepared for ICCV 2025 Challenge.  
It includes the implementation of the **SL-GCN (Skeleton-based Graph Convolutional Network)** model for gesture classification.

## 📁 Project Structure

- `SL-GCN/` – SL-GCN model (taken from SAM-SLR)
- `Pose_estimation/` – Scripts used for pose/keypoint preprocessing



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
