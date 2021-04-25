import cv2
import numpy as np 
import pickle
import os

thresh = 0

in_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_train_color\\1"
im_gray = cv2.imread(f'{in_dir}\\1.png', cv2.IMREAD_GRAYSCALE)

im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

print(type(im_bw))  		# ----------------> <class 'numpy.ndarray'>
print(np.unique(im_bw))		# ----------------> [  0 255]


save_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\RBM"
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

# ------------------------ Train Data ----------------------------------------

train_data = []

# No of videos
for i in range(1000):
	in_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_train_color\\{i}"

	# Number of frames in each video
	for j in range(100):
		# Read image as GrayScale
		im_gray = cv2.imread(f'{in_dir}\\{j}.png', cv2.IMREAD_GRAYSCALE)

		# Convert to binary by thresholding
		im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

		# Add to TrainData
		train_data.append(im_bw/255)



train_data = np.array(train_data)
print("\n\nTrain data : ", train_data.shape)

train_data = train_data.reshape(train_data.shape[0], -1)
print("Train data reshaped: ", train_data.shape)

# Save data to file
pickle.dump(train_data, open(save_dir+'\\train.pkl' , 'wb' ) )
print("Train data saved to file")

# ------------------------ Test Data ----------------------------------------

test_data = []

# No of videos
for i in range(300):
	in_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_test_color\\{i}"

	# Number of frames in each video
	for j in range(100):
		# Read image as GrayScale
		im_gray = cv2.imread(f'{in_dir}\\{j}.png', cv2.IMREAD_GRAYSCALE)

		# Convert to binary by thresholding
		im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

		# Add to testData
		test_data.append(im_bw/255)


test_data = np.array(test_data)
print("\n\nTest data : ", test_data.shape)

test_data = test_data.reshape(test_data.shape[0], -1)
print("Test data reshaped: ", test_data.shape)

# Save data to file
pickle.dump(test_data, open(save_dir+'\\test.pkl' , 'wb' ) )
print("Test data saved to file")