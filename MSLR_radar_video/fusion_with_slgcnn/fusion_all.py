# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 02:53:05 2025

@author: sb3682
"""

import pandas as pd
import numpy as np
import os

# Define weights
w_slg = 0.75      # for 2-model fusions
w_md  = 0.25
w_vid = 0.25

# Define output directory
output_dir = r'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/fusion_with_slgcnn/predictions'
os.makedirs(output_dir, exist_ok=True)

# Process both test and validation splits
for split in ['test', 'val']:
    # Load logits
    md_df     = pd.read_excel(
        rf'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/results/logit_md_{split}.xlsx'
    )
    video_df  = pd.read_excel(
        rf'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/results/logit_video_{split}.xlsx'
    )
    slgcnn_df = pd.read_csv(
        rf'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/kaggle_comp/fusion/multimodal_italian_sign_language_recognition/results/logit_slgcnn_{split}.csv'
    )
    
    # Extract IDs and logit matrices
    ids           = md_df['id']
    logits_md     = md_df.drop(columns=['id']).values
    logits_video  = video_df.drop(columns=['id']).values
    logits_slg    = slgcnn_df.drop(columns=['id']).values
    
    # 1) MD + SLGCNN (MD=0.25, SLGCNN=0.75)
    logits_md_slg = w_md * logits_md + w_slg * logits_slg
    pred_md_slg   = np.argmax(logits_md_slg, axis=1)
    pd.DataFrame({'id': ids, 'pred': pred_md_slg}) \
      .to_csv(os.path.join(output_dir, f'fusion_md_slgcnn_{split}.csv'), index=False)
    
    # 2) Video + SLGCNN (Video=0.25, SLGCNN=0.75)
    logits_vid_slg = w_vid * logits_video + w_slg * logits_slg
    pred_vid_slg   = np.argmax(logits_vid_slg, axis=1)
    pd.DataFrame({'id': ids, 'pred': pred_vid_slg}) \
      .to_csv(os.path.join(output_dir, f'fusion_video_slgcnn_{split}.csv'), index=False)
    
    # 3) SLGCNN + MD + Video (SLGCNN=0.5, MD=0.25, Video=0.25)
    logits_all = 0.5 * logits_slg + w_md * logits_md + w_vid * logits_video
    pred_all   = np.argmax(logits_all, axis=1)
    pd.DataFrame({'id': ids, 'pred': pred_all}) \
      .to_csv(os.path.join(output_dir, f'fusion_slgcnn_md_video_{split}.csv'), index=False)
    
    print(f'Fusion CSV files generated for {split}:')
    print(f' - {output_dir}/fusion_md_slgcnn_{split}.csv')
    print(f' - {output_dir}/fusion_video_slgcnn_{split}.csv')
    print(f' - {output_dir}/fusion_slgcnn_md_video_{split}.csv')
