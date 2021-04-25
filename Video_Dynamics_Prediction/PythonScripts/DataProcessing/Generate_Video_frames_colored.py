import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

from PIL import Image, ImageOps

# ------------------------ Train Data ----------------------------------------

with open('C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_train.pkl', 'rb') as f:
    data = pickle.load(f)

# print(type(data))

# print(data.keys())
# print(data["X"].shape)

# print(data["X"][0,0].shape)

# cv2.imshow("image", data["X"][0,0])
# cv2.waitKey()
# cv2.imwrite("C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\filename.png", data["X"][0,0]*255)
# print(np.max(data["X"][0,0]))
# print(np.min(data["X"][0,0]))


for i in range(data["X"].shape[0]):
	directory = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_train_color\\{i}"
	if not os.path.exists(directory):
		os.makedirs(directory)

	for j in range(data["X"].shape[1]):
		cv2.imwrite(f"{directory}\\{j}.png", data["X"][i,j]*255)


# ------------------------ Test Data ----------------------------------------

with open('C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_test.pkl', 'rb') as f:
    data = pickle.load(f)

# print(type(data))

# print(data.keys())
# print(data["X"].shape)

# print(data["X"][0,0].shape)

# cv2.imshow("image", data["X"][0,0])
# cv2.waitKey()
# cv2.imwrite("C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\filename.png", data["X"][0,0]*255)
# print(np.max(data["X"][0,0]))
# print(np.min(data["X"][0,0]))


for i in range(data["X"].shape[0]):
	directory = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_test_color\\{i}"
	if not os.path.exists(directory):
		os.makedirs(directory)

	for j in range(data["X"].shape[1]):
		cv2.imwrite(f"{directory}\\{j}.png", data["X"][i,j]*255)