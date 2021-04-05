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


class HMM():

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


	# Plot Gaussian Contours
	def	plot_Gaussian(self, ax, y, sigma, mu):
		x1 = np.linspace(start=min(y[:,0]), stop=max(y[:,0]), num=150)
		x2 = np.linspace(start=min(y[:,1]), stop=max(y[:,1]), num=150)
		X,Y = np.meshgrid(x1, x2)

		# Probability Density
		rv = multivariate_normal(mu, sigma)
		pos = np.empty(X.shape + (2,))
		pos[:, :, 0] = X
		pos[:, :, 1] = Y
		pdf = rv.pdf(pos)

		ax.contour(X, Y, pdf)

	# FUnction for initializing parameters
	def initialize_params(self, Y):

		print("\n\n\n", "-"*60, sep="")
		print("\t\tInitializing params")
		print("-"*60, "\n\n")

		# Split data into chunks
		data = Y
		np.random.shuffle(np.array(data))
		data = np.array_split(data, self.num_class)

		# pi : probability for categorical distribution for latent variable X : P(X=i)
		# initialized uniformly
		self.pi = np.ones(self.num_class)*1.0/(self.num_class)

		# A : Transition probability matrix for categorical distribution for latent variable X : p(X_n|X_{n-1}) = Categ[AX_{n-1}]
		# initialized uniformly
		self.A = np.ones((self.num_class, self.num_class))*1.0/(self.num_class)

		# C : Matrix of mean of Y|X : P(Y|X)~N(CX, sigma_x)
		# initialized using sample means from random data split
		self.C = [np.mean(d, axis=0) for d in data]

		# Sigma : Covariance matrix of y|x
		# initialized using sample means from random data split
		self.Sigma = [np.cov(d.T) for d in data]
		self.Sigma_inv = [np.linalg.pinv(S) for S in self.Sigma]

		# k = 1/2pi*det(Sigma)
		self.k = [1.0/((((2*math.pi)**Y.shape[1])*np.linalg.det(S))**0.5) for S in self.Sigma]

		# Posterior Probability
		self.q_x_y = np.zeros((len(Y), self.num_class))

		print("pi : ", self.pi)
		print("\n\nA : \n", self.A)
		print("\n\n\nC : \n", self.C)
		print("\n\n\nSigma : \n", self.Sigma)


	# Filtering
	def Filtering(self, y):

		time_update = np.zeros((y.shape[0], self.num_class))
		measure_update = np.zeros((y.shape[0], self.num_class))

		# q(X_n|X_{n-1})
		q_xn_xn_1 = self.A

		# 1st measurement update
		for i in range(self.num_class):
			# q(y_1|x_1)
			# y_cap = np.array(y[0,:]).reshape((y.shape[1], 1))
			q_x1 = self.pi[i]
			q_y1_given_x1 = self.k[i]*np.exp(-0.5*(y[0,:] - self.C[i]).T @ self.Sigma_inv[i] @ (y[0,:] - self.C[i]))

			# q(x_1|y_1)
			measure_update[0, i] = q_x1 * q_y1_given_x1

		measure_update[0, :] = measure_update[0, :]/np.sum(measure_update[0, :], axis=0)

		# time & measurement update for all n-{1}
		for row in range(1, y.shape[0]):
			for col in range(self.num_class):
				# q(y_n|x_n)
				q_yn_given_xn = self.k[col]*np.exp(-0.5*(y[row,:] - self.C[col]).T @ self.Sigma_inv[col] @ (y[row,:] - self.C[col]))
				# Time Update
				for i in range(self.num_class):
					time_update[row, col] += measure_update[row-1, i] * q_xn_xn_1[col, i] 

				# Measurement Update
				measure_update[row, col] = time_update[row, col] *q_yn_given_xn

			measure_update[row, :] = measure_update[row, :]/np.sum(measure_update[row, :], axis=0)

		self.q_xn_given_yn = measure_update


	# Smoothing
	def Smoothing(self, y):

		future_cond = np.zeros((y.shape[0], self.num_class, self.num_class))
		self.q_x_x_y = np.zeros((y.shape[0]-1, self.num_class, self.num_class))
		backward_step = np.zeros((y.shape[0], self.num_class))

		# q(X_n|X_{n-1})
		q_xn_xn_1 = self.A

		# P(X_n|y_1, ...y_N) initialised separately
		backward_step[-1, :] = self.q_xn_given_yn[-1, :]

		# Future conditioning and backward step for all n-{N}
		for row in range(y.shape[0]-2, -1, -1):
			y_cap = np.array(y[row+1,:]).reshape((y.shape[1], 1))
			
			for i in range(self.num_class):	
				for col in range(self.num_class):
					# Future Conditioning
					future_cond[row, col, i] = self.q_xn_given_yn[row, col] * q_xn_xn_1[i, col]

				future_cond[row, :, i] = future_cond[row, :, i]/np.sum(future_cond[row, :, i], axis=0)

			for col in range(self.num_class):
				for i in range(self.num_class):	
					# Backward step
					backward_step[row, col] += backward_step[row+1, i] * future_cond[row, col, i]

					self.q_x_x_y[row, col, i] = backward_step[row+1, i] * future_cond[row, col, i]

		self.q_x_y = backward_step


	def Maximization(self, Y):

		# Updating pi
		self.pi = np.mean(self.q_x_y, axis=0)

		# Updating A
		# for i in range(self.num_class):
		# 	for j in range(self.num_class):
		# 		self.A[i,j] = np.sum(self.q_x_x_y[i,j])

		# for j in range(self.num_class):
		# 		self.A[:,j] = self.A[:,j]/np.sum(self.A[:,j])

		self.A = np.sum(self.q_x_x_y, axis=0)
		self.A = self.A/np.sum(self.A, axis=0)

		n = self.q_x_y.shape[0]

		# Updating Sigma
		self.Sigma = [np.cov(Y.T, aweights=self.q_x_y[:,i]) for i in range(self.num_class)]
		self.Sigma_inv = [np.linalg.pinv(S) for S in self.Sigma]
		self.k = [1.0/((((2*math.pi)**Y.shape[1])*np.linalg.det(S))**0.5) for S in self.Sigma]

		# Updating C
		self.C = [np.sum(self.q_x_y[:,i].reshape(n,1)*Y, axis=0)/(np.sum(self.q_x_y[:,i])) for i in range(self.num_class)]


	def Expectation_Maximization(self, Y):

		# Initialize parameters using data
		self.initialize_params(Y)

		print("\n\n\n", "-"*60, sep="")
		print("\t\tExpectation Maximization")
		print("-"*60, "\n\n")

		print("Running.....")

		# Iterations
		for e in range(self.epochs):
			self.Filtering(Y)
			self.Smoothing(Y)
			# self.HMM_alphaRecusrion(Y)
			# self.HMM_gammaRecusrion(Y)
			self.Maximization(Y)

			# print(self.q_xn_given_yn)

		self.Filtering(Y)
		self.Smoothing(Y)
		# self.HMM_alphaRecusrion(Y)
		# self.HMM_gammaRecusrion(Y)

		print("Finished.....")

		print("\n\n\n", "-"*60, sep="")
		print("\t\tFinal Parameters")
		print("-"*60, "\n\n")
		print("pi : ", self.pi)
		print("\n\nA : \n", self.A)
		print("\n\n\nC : \n", self.C)
		print("\n\n\nSigma : \n", self.Sigma)

		fig, ax = plt.subplots()
		ax.scatter(Y[:,0],Y[:,1], marker='o', c='r')
		self.plot_Gaussian(ax, Y, self.Sigma[0], self.C[0])
		self.plot_Gaussian(ax, Y, self.Sigma[1], self.C[1])
		self.plot_Gaussian(ax, Y, self.Sigma[2], self.C[2])

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
	hmm = HMM(config)

	# Train GMM using Expectation Maximization 
	predictions = hmm.Expectation_Maximization(Y)

	print(X)

	print("\n\n\n")
	# print(predictions)

	data["predictions"] = 2-predictions
	print(data["predictions"].values)

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

	print("\n\n\nAccuracy of HMM:", Accuracy(data[[0]].values, data[["predictions"]].values))

			
	data.to_csv(args.out_file, sep="\t")
	print("\n\n Outputs saved in {}".format(args.out_file))
	print("\n\n\t\tEnd of program")

	



if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", dest='data_file', type=str, default="HMMdata.csv",  action='store', help="csv file containing the data")
	parser.add_argument("--out_file", dest='out_file', type=str, default="EM_HMM_output.tsv",  action='store', help="tsv file containing the output predictions")
	parser.add_argument("--label_idx", dest='label_idx', type=int, default=0,  action='store', help="column index of true labels in data (assumed to be integer valued, starting from 0)")
	parser.add_argument("--epochs", dest='epochs', type=int, default=25,  action='store', help="no. of epochs over the data")
	args = parser.parse_args()
	main(args)