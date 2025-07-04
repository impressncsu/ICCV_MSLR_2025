# SLGCN Based Isolated Italian Sign Language Recognition
## Data preparation
1. copy/place the generated wholebody keypoint npy files to SL-GCN/wholepose_test_npy folder for the next step
2. Run the following code to prepare the data for GCN.

        cd data_gen/
        python sign_gendata_test.py
        python gen_bone_data_test.py
        python gen_motion_test.py
        
## Predicting Using SLGCN
        
### Download pretrained weights  for the SLGCN module
1. download the save_model.zip folder 

 https://drive.google.com/file/d/14kS0_yFGE0217vzck71Ni_Z2_TsEFHvV/view?usp=sharing
 
2. Extract inside SL-GCN (ICCV2025_MSLR/SL-GCN/save_models after extraction)
        
3. python run_all_configs.py

4. python run_fusion.py


