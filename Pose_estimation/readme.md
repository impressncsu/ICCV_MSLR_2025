## 133-point whole-body pose extraction using HRNet

1. Have the test videos in this format SAMPLE_x.mp4 in the desired directory

2. download pretrained model based on HRNET

https://drive.google.com/file/d/1f_c3uKTDQ4DR3CrwMSI8qdsTKJvKVt7p/view?usp=sharing

4. place it inside Pose_estimation folder

4. Change the input_path and output_dir in demo.py based on the location of mp4 files and the desired ouput location to save the keypoint .npy files

5. run python demo.py

6. copy/place the generated wholebody keypoint npy files to SL-GCN/wholepose_test_npy folder for the next step

7. Manually create the wholepose_test_npy folder if necessary


