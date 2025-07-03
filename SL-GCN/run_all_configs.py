import subprocess
import os
from pathlib import Path

def run_all_configs():
    base_dir = Path("config/sign/test/fold_check")
    subfolders = ["joint_folds", "bone_folds", "joint_m_folds", "bone_m_folds"]

    for folder in subfolders:
        config_dir = base_dir / folder
        yaml_files = sorted(config_dir.glob("*.yaml"))

        for yaml_file in yaml_files:
            print(f"\n🚀 Running: {yaml_file}\n")
            result = subprocess.run(
                ["python", "main.py", "--config", str(yaml_file)],
                capture_output=True,
                text=True
            )

            print(result.stdout)
            if result.returncode != 0:
                print(f"❌ Error in {yaml_file}:\n{result.stderr}")

if __name__ == "__main__":
    run_all_configs()
