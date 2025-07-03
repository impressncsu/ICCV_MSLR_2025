import os
import pickle
import numpy as np
import pandas as pd

# Paths
base_dir = './work_dir'  
save_dir = '../Fusion' 
os.makedirs(save_dir, exist_ok=True)

# Streams and weights
identifiers = ['testing_joint', 'testing_jm', 'testing_bone', 'testing_bm']
weights = {
    'testing_joint': 0.5,
    'testing_jm': 0.3,
    'testing_bone': 0.1,
    'testing_bm': 0.1
}


stream_dataframes = {}

for identifier in identifiers:
    print(f"\nProcessing: {identifier}")
    pkl_files = [
        os.path.join(base_dir, f"{identifier}_fold{fold}", "eval_results", "best_acc.pkl")
        for fold in range(5)
    ]

    data_list = []
    sample_names = None
    for i, pkl_file in enumerate(pkl_files):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        current_keys = list(data.keys())
        if i == 0:
            sample_names = current_keys
        else:
            if sample_names != current_keys:
                raise ValueError(f"Sample names mismatch in {pkl_file}")
        data_list.append(data)

    num_samples = len(sample_names)
    num_classes = len(data_list[0][sample_names[0]])
    averaged_matrix = np.zeros((num_samples, num_classes), dtype=np.float64)

    for i, name in enumerate(sample_names):
        scores = [np.array(d[name], dtype=np.float64) for d in data_list]
        averaged_matrix[i] = np.mean(scores, axis=0)

    logit_columns = [f"logit{i}" for i in range(num_classes)]
    df = pd.DataFrame(averaged_matrix, columns=logit_columns)
    df.insert(0, "id", sample_names)
    df['id'] = df['id'].str.replace('SAMPLE_', '', regex=False).astype(int)
    df = df.sort_values(by='id').reset_index(drop=True)
    stream_dataframes[identifier] = df

base_ids = stream_dataframes['testing_joint']['id'].values
for key in weights:
    ids = stream_dataframes[key]['id'].values
    if not np.array_equal(base_ids, ids):
        raise ValueError(f"ID mismatch or order mismatch in {key}")

weighted_logits = np.zeros_like(
    stream_dataframes['testing_joint'].iloc[:, 1:].values, dtype=np.float64
)
for key, weight in weights.items():
    logits = stream_dataframes[key].iloc[:, 1:].values
    weighted_logits += weight * logits

logit_columns = [f"logit{i}" for i in range(weighted_logits.shape[1])]
final_df = pd.DataFrame(weighted_logits, columns=logit_columns)
final_df.insert(0, 'id', base_ids)

final_csv_path = os.path.join(save_dir, 'slgcn_logits.csv')

final_df.to_csv(final_csv_path, index=False)
print(f"SLGCN logits saved to: {final_csv_path}")
