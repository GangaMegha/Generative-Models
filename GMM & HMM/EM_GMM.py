import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import math
from scipy.stats import multivariate_normal

# Read data from file
def loadData(filename):
	data = pd.read_csv(filename, sep=',', header=None)
	return data

# Compute accuract in %
def Accuracy(true, pred):
	return np.mean(true==pred)*100


class GMM():

	def __init__(self, config):

		print("\n\n\n\n\n\n", "*"*60, sep="")
		print("\t\tGaussian Mixture Models\n")
		print("\t    Latent Variable Density Estimation")
		print("*"*60, "\n\n")

		# Updating dictionary
		self.__dict__.update(config)

		# Categorical Latent Variable
		self.X = np.identity(self.num_class)

		# epsilon to prevent division by zero error
		self.epsilon = 1e-10


	# FUnction for initializing parameters
	def initialize_params(self, Y):

		print("\n\n\n", "-"*60, sep="")
		print("Step 1 : \tInitializing params")
		print("-"*60, "\n\n")

		# Split data into chunks
		data = Y
		np.random.shuffle(np.array(data))
		data = np.array_split(data, self.num_class)

		# pi : probability for categorical distribution for latent variable X : P(X=i)
		# initialized uniformly
		self.pi = np.ones(self.num_class)*1.0/(self.num_class)

		# C : Matrix of mean of Y|X : P(Y|X)~N(CX, sigma_x)
		# initialized using sample means from random data split
		self.C = [np.mean(d, axis=0) for d in data]

		# Sigma : Covariance matrix of y|x
		# initialized using sample means from random data split
		self.Sigma = [np.cov(d.T) for d in data]
		self.Sigma_inv = [np.linalg.pinv(S) for S in self.Sigma]

		# k = 1/2pi*det(Sigma)
		self.k = [1.0/(2*math.pi*np.linalg.det(S)) for S in self.Sigma]

		# Posterior Probability
		self.q_x_y = np.zeros((len(Y), self.num_class))

		print("pi : ", self.pi)
		print("\n\n\nC : \n", self.C)
		print("\n\n\nSigma : \n", self.Sigma)


	# Gaussian Mixture Model Inference			
	def Expectation(self, Y):

		# Computing posterior probability
		for row, y in enumerate(Y):
			for i in range(self.num_class):
				q_y_x = self.k[i]*np.exp(-0.5*(y - self.C[i]).T @ self.Sigma_inv[i] @ (y - self.C[i]))
				q_x = self.pi[i]
				self.q_x_y[row, i] = q_y_x * q_x

			self.q_x_y[row,:] = self.q_x_y[row,:]/(np.sum(self.q_x_y[row,:]))


	def Maximization(self, Y):

		# Updating pi
		self.pi = np.mean(self.q_x_y, axis=0)

		n = self.q_x_y.shape[0]

		# Updating Sigma
		self.Sigma = [np.cov(Y.T, aweights=self.q_x_y[:,i]) for i in range(self.num_class)]
		self.Sigma_inv = [np.linalg.pinv(S) for S in self.Sigma]
		self.k = [1.0/(2*math.pi*np.linalg.det(S)) for S in self.Sigma]

		# Updating C
		self.C = [np.sum(self.q_x_y[:,i].reshape(n,1)*Y, axis=0)/(np.sum(self.q_x_y[:,i])) for i in range(self.num_class)]


	def Expectation_Maximization(self, Y):

		# Initialize parameters using data
		self.initialize_params(Y)

		print("\n\n\n", "-"*60, sep="")
		print("\t\tExpectation Maximization")
		print("-"*60, "\n\n")

		# Iterations
		for e in range(self.epochs):
			self.Expectation(Y)
			self.Maximization(Y)

		self.Expectation(Y)

		return(np.argmax(self.q_x_y, axis=1))
		


def main(args):
	config = vars(args)

	# Load data
	data = loadData(args.data_file)

	# Infer classes
	config["num_class"] = len(data[args.label_idx].unique())

	# Label
	X = data[args.label_idx].values

	# Data
	Y = data.drop(args.label_idx, axis=1).values

	# Create GMM instance
	gmm = GMM(config)

	# Train GMM using Expectation Maximization 
	predictions = gmm.Expectation_Maximization(Y)

	print(X)

	print("\n\n\n")
	print(predictions)

	data["predictions"] = 2-predictions

	# Plot data as a scatter plot
	fig, ax = plt.subplots(1,2)
	ax[0].scatter(data[data[args.label_idx]==0][1], data[data[args.label_idx]==0][2], marker='o', c='b', label="x=0")
	ax[0].scatter(data[data[args.label_idx]==1][1], data[data[args.label_idx]==1][2], marker='o', c='g', label="x=1")
	ax[0].scatter(data[data[args.label_idx]==2][1], data[data[args.label_idx]==2][2], marker='o', c='r', label="x=2")
	ax[0].legend()

	ax[1].scatter(data[data["predictions"]==0][1], data[data["predictions"]==0][2], marker='o', c='b', label="x=0")
	ax[1].scatter(data[data["predictions"]==1][1], data[data["predictions"]==1][2], marker='o', c='g', label="x=1")
	ax[1].scatter(data[data["predictions"]==2][1], data[data["predictions"]==2][2], marker='o', c='r', label="x=2")
	ax[1].legend()
	plt.show()

	print("\n\n\nAccuracy of GMM:", Accuracy(data[[0]].values, data[["predictions"]].values))

			
	data.to_csv(args.out_file, sep="\t")
	print("\n\n Outputs saved in {}".format(args.out_file))
	print("\n\n\t\tEnd of program")

	



if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", dest='data_file', type=str, default="HMMdata.csv",  action='store', help="csv file containing the data")
	parser.add_argument("--out_file", dest='out_file', type=str, default="EM_GMM_output.tsv",  action='store', help="tsv file containing the output predictions")
	parser.add_argument("--label_idx", dest='label_idx', type=int, default=0,  action='store', help="column index of true labels in data (assumed to be integer valued, starting from 0)")
	parser.add_argument("--epochs", dest='epochs', type=int, default=500,  action='store', help="no. of epochs over the data")
	args = parser.parse_args()
	main(args)