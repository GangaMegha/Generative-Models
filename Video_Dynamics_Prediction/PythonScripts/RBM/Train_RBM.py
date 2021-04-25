import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os

from RBM import RBM 

def read_data(data_dir):
	train = pickle.load( open(data_dir +'\\train.pkl' , 'rb' ))
	test = pickle.load( open(data_dir +'\\test.pkl' , 'rb' ))

	return train, test

def split_data(train):
	# Split data into Train, Val, Test and flatten the images
	frac = 0.15

	n = int(frac*(train.shape[0]))
	val = train[:n]
	train = train[n:]

	return train, val

def plot_error(train_loss, val_loss, out_dir):
	fig = plt.figure()
	plt.plot(train_loss, c='r', label="Train")
	plt.plot(val_loss, c='g', label="Val")
	plt.legend()
	plt.title("Reconstruction Error")
	plt.xlabel("Epoch")
	plt.ylabel("Error")
	plt.savefig(out_dir + "Error_plot.png")


def reconstruct_images(rbm, test, dim, out_dir):

	# Shuffling data inorder to get independent images
	test = np.random.permutation(test)

	print("\n\nRunning image reconstruction on 6 images from test data...")

	fig1 = plt.figure()
	fig1.suptitle('Reconstruction on test images', fontsize=16)

	count = 0

	for i in range(1, 12, 2):

		# Displaying Original Image
		plt.subplot(6,2,i)
		plt.tight_layout()
		plt.imshow(test[count].reshape(dim), cmap='gray', interpolation='none')
		plt.title("Original")
		plt.xticks([])
		plt.yticks([])

		# Displaying Reconstructed Image
		plt.subplot(6,2,i+1)
		plt.tight_layout()

		# 1 step sampling
		img = rbm.reconstruct_image(test[count].T.reshape((dim[0]*dim[1],1)))

		plt.imshow(img.reshape(dim), cmap='gray', interpolation='none')
		plt.title("Reconstructed")
		plt.xticks([])
		plt.yticks([])

		count += 1

	plt.savefig(out_dir + "reconstructed_test_images.png")


def main(args):
	train, test = read_data(args.in_dir)
	train, val = split_data(train)

	print('\n\nTrain: ', train.shape)
	print('Val: ', val.shape)
	print('Test:  ', test.shape)

	# RBM object
	rbm = RBM(args.num_hidden, val.shape[1], args.lr, args.n, args.batch_size, args.epochs)

	# Train RBM
	train_loss, val_loss = rbm.Train(train, val)

	# Create output dir if it doesn't exist
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	# Plot error
	plot_error(train_loss, val_loss, args.out_dir)

	# Performance on Test set
	error_test = rbm.reconstruction_error(test.T)

	print("\n\n\nReconstruction error...\n")
	print("Train : ", train_loss[-1])
	print("Val : ", val_loss[-1])
	print("Test : ", error_test)

	# For viewing reconstruction
	reconstruct_images(rbm, test, (args.image_height, args.image_width), args.out_dir)

	# Saving the model learned weights
	pickle.dump( [rbm.W, rbm.b_h, rbm.b_v] , open(args.out_dir+'\\weights.pkl' , 'wb' ) )
	print(f"\n\nRBM weights saved in {args.out_dir}\\weights.pkl")

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_dir', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\RBM\\", help='folder containing train.pkl and test.pkl')
	parser.add_argument('--out_dir', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\RBM\\", help='Test file')
	parser.add_argument('--num_hidden', type=int, default=128, help='No. of hidden units in RBM')
	parser.add_argument('--epochs', type=int, default=1000, help='No. of epochs to train')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for gradient descent')
	parser.add_argument('--batch_size', type=int, default=100, help='Mini batch size during training')
	parser.add_argument('--n', type=int, default=1, help='No. of Gibbs sampling steps for contrastive divergence')
	parser.add_argument('--image_height', type=int, default=32, help='For reshaping image to see the reconstruction')
	parser.add_argument('--image_width', type=int, default=32, help='For reshaping image to see the reconstruction')
	args = parser.parse_args()
	main(args)