import cv2
import os

# im_gray = cv2.imread('C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\filename.png', cv2.IMREAD_GRAYSCALE)
# cv2.imshow("image gray", im_gray)
# cv2.waitKey()
# thresh = 0
# im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("image binary", im_bw)
# cv2.waitKey()
# cv2.imwrite('C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\filename_binary.png', im_bw)

thresh = 0


# ------------------------ Train Data ----------------------------------------

# No of videos
for i in range(1000):
	in_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_train_color\\{i}"
	out_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_train\\{i}"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# Number of frames in each video
	for j in range(100):
		# Read image as GrayScale
		im_gray = cv2.imread(f'{in_dir}\\{j}.png', cv2.IMREAD_GRAYSCALE)

		# Convert to binary by thresholding
		im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

		# Write binary image to file
		cv2.imwrite(f"{out_dir}\\{j}.png", im_bw)


# ------------------------ Test Data ----------------------------------------

# No of videos
for i in range(300):
	in_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_test_color\\{i}"
	out_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\billiards_test\\{i}"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# Number of frames in each video
	for j in range(100):
		# Read image as GrayScale
		im_gray = cv2.imread(f'{in_dir}\\{j}.png', cv2.IMREAD_GRAYSCALE)

		# Convert to binary by thresholding
		im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

		# Write binary image to file
		cv2.imwrite(f"{out_dir}\\{j}.png", im_bw)