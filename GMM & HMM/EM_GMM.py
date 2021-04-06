import numpy as np 
import math
from scipy.stats import multivariate_normal

class GMM():

	def __init__(self, config):

		print("\n\n\n\n\n\n", "*"*60, sep="")
		print("\t\tGaussian Mixture Models\n")
		print("\t    Latent Variable Density Estimation")
		print("*"*60, "\n\n")

		# Updating dictionary
		self.__dict__.update(config)


	# FUnction for initializing parameters
	def initialize_params(self, Y):

		print("\n\n\n", "-"*60, sep="")
		print("\t\tInitializing params")
		print("-"*60, "\n\n")

		# Split data into chunks
		data = Y
		np.random.shuffle(np.array(data))
		data = np.array_split(data, self.num_class)

		# Number of observations (n = 1 to N)
		self.N = Y.shape[0]

		# Dimention of data
		self.D = Y.shape[1]

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
		self.k = [1.0/((((2*math.pi)**self.D)*np.linalg.det(S))**0.5) for S in self.Sigma]

		# Posterior Probability : q(X|Y)
		self.q_x_y = np.zeros((self.N, self.num_class))
		self.q_x_y_prev = np.zeros((self.N, self.num_class))

		# log_joint = -log q(X, Y)
		self.log_joint = np.zeros((self.N, self.num_class))

		print("pi : ", self.pi)
		print("\n\n\nC : \n", self.C)
		print("\n\n\nSigma : \n", self.Sigma)


	# Gaussian Mixture Model Inference			
	def Expectation(self, Y):

		# Computing posterior probability
		self.q_x_y_prev = self.q_x_y.copy()

		for row, y in enumerate(Y):
			for i in range(self.num_class):
				q_y_x = self.k[i]*np.exp(-0.5*(y - self.C[i]).T @ self.Sigma_inv[i] @ (y - self.C[i]))
				q_x = self.pi[i]

				self.q_x_y[row, i] = q_y_x * q_x
				self.log_joint[row, i] = - np.log(self.q_x_y[row, i]) 

			self.q_x_y[row,:] = self.q_x_y[row,:]/(np.sum(self.q_x_y[row,:]))


	def Maximization(self, Y):

		# Updating pi
		self.pi = np.mean(self.q_x_y, axis=0)

		# Updating Sigma
		self.Sigma = [np.cov(Y.T, aweights=self.q_x_y[:,i]) for i in range(self.num_class)]
		self.Sigma_inv = [np.linalg.pinv(S) for S in self.Sigma]
		self.k = [1.0/((((2*math.pi)**self.D)*np.linalg.det(S))**0.5) for S in self.Sigma]

		# Updating C
		self.C = [np.sum(self.q_x_y[:,i].reshape(self.N,1)*Y, axis=0)/(np.sum(self.q_x_y[:,i])) for i in range(self.num_class)]

			
	def FullCrossEntropy(self):
		# <-log q(X,Y)>_{p(Y)q(X|Y)}
		self.Entropy = np.mean(np.sum(self.log_joint * self.q_x_y, axis=1), axis=0)
		self.err = np.mean((self.q_x_y-self.q_x_y_prev)**2)

	def Expectation_Maximization(self, Y):

		# Initialize parameters using data
		self.initialize_params(Y)

		print("\n\n\n", "-"*60, sep="")
		print("\t\tExpectation Maximization")
		print("-"*60, "\n\n")

		print("Running.....")

		self.Expectation(Y)

		# Iterations
		for e in range(self.epochs):
			self.Maximization(Y)
			self.Expectation(Y)
			self.FullCrossEntropy()

			# Stopping Criteria
			print(f"Epoch {e+1} ====> Full Cross Entropy : {self.Entropy}, Err : {self.err}")
			if self.err < 2.5e-7:
				break 


		print("Finished.....")

		print("\n\n\n", "-"*60, sep="")
		print("\t\tFinal Parameters")
		print("-"*60, "\n\n")
		print("pi : ", self.pi)
		print("\n\n\nC : \n", self.C)
		print("\n\n\nSigma : \n", self.Sigma)

		return self.q_x_y, self.Sigma, self.C