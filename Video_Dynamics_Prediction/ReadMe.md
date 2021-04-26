# Video Generation :​ Capturing Motion Dynamics​

### Objective : 
Use a generative model to predict motion dynamics ​

For more information, please check the [PPT](https://purdue0-my.sharepoint.com/:p:/g/personal/gmeghana_purdue_edu/ERBG7qfqsHFCg8H2FN1BNPkB1FwD7v_Z-p4fcFtcSu2gBw?e=Px10PY)

### Instructions for testing

##### 1. Generate Data
Execute :
`create_data.py ----num_obj 3`

num_obj indicated the number of balls that will interact in the Billiards env.

##### 2. Data Processing
Go into Video_Dynamics_Prediction/PythonScripts/DataProcessing/ and run the following :
1. Generate_Video_frames_colored.py
2. Convert_img_to_binary.py
3. GenerateRBM_Train_Test_data.py


##### 3. Train RBM
Go into Video_Dynamics_Prediction/PythonScripts/RBM/ and run Train_RBM.py with the required config.


##### 4. Train Dynamics prediction model
Go into Video_Dynamics_Prediction/PythonScripts/Dynamics/ and run the following :
1. Generate_VideoFrames_Latent_z.py
2. GenerateHidden_FromTrainedRBM.py
and the run Predict_Dinamics.py with the required config.


##### 5. Train State prediction model
Go into Video_Dynamics_Prediction/PythonScripts/Generate_Hidden/ and run the following :
1. Generate_VideoFrames_Latent_z.py
2. GenerateHidden_FromTrainedRBM.py
and the run Predict_Hidden.py with the required config.


##### 6. Video Generation
Go into Video_Dynamics_Prediction/PythonScripts/Video_Prediction/ and run Predict_video_frames.py