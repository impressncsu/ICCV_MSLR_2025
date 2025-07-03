import pandas as pd
import numpy as np
from scipy.special import softmax
import os

def main():
    print("🔄 Loading logits files...")

  
    slgcn_df = pd.read_csv('Fusion/slgcn_logits.csv')
    md_df = pd.read_excel('MSLR_radar_video/results/logit_md_test.xlsx')
    video_df = pd.read_excel('MSLR_radar_video/results/logit_video_test.xlsx')

 
    slgcn_df = slgcn_df.sort_values('id').reset_index(drop=True)
    md_df = md_df.sort_values('id').reset_index(drop=True)
    video_df = video_df.sort_values('id').reset_index(drop=True)

    
    assert np.array_equal(slgcn_df['id'], md_df['id']), "❌ ID mismatch: SLGCN vs MD"
    assert np.array_equal(slgcn_df['id'], video_df['id']), "❌ ID mismatch: SLGCN vs Video"

    print("IDs aligned across all sources.")

    
    slgcn_logits = slgcn_df.iloc[:, 1:].values
    md_logits = md_df.iloc[:, 1:].values
    video_logits = video_df.iloc[:, 1:].values

    fused_logits = 0.5 * slgcn_logits + 0.25 * md_logits + 0.25 * video_logits
    fused_probs = softmax(fused_logits, axis=1)
    preds = np.argmax(fused_probs, axis=1)

    
    result_df = pd.DataFrame({
        'id': slgcn_df['id'],
        'pred': preds
    })

    # Ensure output directory exists
    output_dir = 'Final_prediction'
    os.makedirs(output_dir, exist_ok=True)

    # Save result
    output_path = os.path.join(output_dir, 'Final_LIS_prediction.csv')
    result_df.to_csv(output_path, index=False)

    print(f"Saved prediction to {output_path}")

if __name__ == "__main__":
    main()
