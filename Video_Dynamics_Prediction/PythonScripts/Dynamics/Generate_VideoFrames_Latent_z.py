import pickle
import cv2
import numpy as np
import os


in_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\"
out_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\Dynamics\\"

if not os.path.exists(out_dir):
	os.makedirs(out_dir)

# ------------------------ Train Data ----------------------------------------

with open(f"{in_dir}\\billiards_train.pkl", 'rb') as f:
    data = pickle.load(f)

# Getting the video frames
frames = []
for i in range(data["y"].shape[0]):

	temp = []

	for j in range(data["y"].shape[1]):
		# Reading the black and white image
		im_bw = cv2.imread(f'{in_dir}\\billiards_train\\{i}\\{j}.png', cv2.IMREAD_GRAYSCALE)
		temp.append(im_bw.reshape(-1)/255)

	frames.append(np.array(temp))

frames = np.array(frames)
print("Frames : ", frames.shape)
print("Frames uniques : ", np.unique(frames))

pickle.dump(frames, open(f"{out_dir}\\train_frames.pkl" , 'wb' ) )


# Getting the latent states
z = []
for i in range(data["y"].shape[0]):

	temp = []

	for j in range(data["y"].shape[1]):
		# Getting the latent dynamics states
		temp.append(data["y"][i,j].reshape(-1))

	z.append(np.array(temp))

# pos in [-1, 1]
z = np.array(z)/10
z[:, : :2] = z[:, : :2]-0.9
print("Latent Dynamics z : ", z.shape)
print("Latent Dynamics z max : ", np.max(z))
print("Latent Dynamics z min : ", np.min(z))

pickle.dump(z, open(f"{out_dir}\\train_z.pkl" , 'wb' ) )


# ------------------------ Test Data ----------------------------------------

with open(f"{in_dir}\\billiards_test.pkl", 'rb') as f:
    data = pickle.load(f)

# Getting the video frames
frames = []
for i in range(data["y"].shape[0]):

	temp = []

	for j in range(data["y"].shape[1]):
		# Reading the black and white image
		im_bw = cv2.imread(f'{in_dir}\\billiards_test\\{i}\\{j}.png', cv2.IMREAD_GRAYSCALE)
		temp.append(im_bw.reshape(-1)/255)

	frames.append(np.array(temp))

frames = np.array(frames)
print("Frames : ", frames.shape)
print("Frames uniques : ", np.unique(frames))

pickle.dump(frames, open(f"{out_dir}\\test_frames.pkl" , 'wb' ) )


# Getting the latent states
z = []
for i in range(data["y"].shape[0]):

	temp = []
	
	for j in range(data["y"].shape[1]):
		# Getting the latent dynamics states
		temp.append(data["y"][i,j].reshape(-1))

	z.append(np.array(temp))

# pos in [-1, 1]
z = np.array(z)/10
z[:, : :2] = z[:, : :2]-0.9
print("Latent Dynamics z : ", z.shape)
print("Latent Dynamics z max : ", np.max(z))
print("Latent Dynamics z min : ", np.min(z))

pickle.dump(z, open(f"{out_dir}\\test_z.pkl" , 'wb' ) )
