import numpy as np 
import math
from scipy.stats import multivariate_normal


class HMM():

	def __init__(self, config):

		print("\n\n\n\n\n\n", "*"*60, sep="")
		print("\t\tHidden Markov Models\n")
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
		self.k = [1.0/((((2*math.pi)**self.D)*np.linalg.det(S))**0.5) for S in self.Sigma]

		# Posterior Probability
		self.q_x_y = np.zeros((len(Y), self.num_class))
		self.q_x_y_prev = np.zeros((len(Y), self.num_class))

		print("pi : ", self.pi)
		print("\n\nA : \n", self.A)
		print("\n\n\nC : \n", self.C)
		print("\n\n\nSigma : \n", self.Sigma)


	# Filtering
	def Filtering(self, y):

		self.q_x_y_prev = self.q_x_y.copy()

		time_update = np.zeros((self.N, self.num_class))
		measure_update = np.zeros((self.N, self.num_class))

		# q(X_n|X_{n-1})
		q_xn_xn_1 = self.A

		# 1st measurement update
		for i in range(self.num_class):
			# q(y_1|x_1)
			q_x1 = self.pi[i]
			q_y1_given_x1 = self.k[i]*np.exp(-0.5*(y[0,:] - self.C[i]).T @ self.Sigma_inv[i] @ (y[0,:] - self.C[i]))

			# q(x_1|y_1)
			measure_update[0, i] = q_x1 * q_y1_given_x1

		measure_update[0, :] = measure_update[0, :]/np.sum(measure_update[0, :], axis=0)

		# time & measurement update for all n-{1}
		for row in range(1, self.N):
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

		future_cond = np.zeros((self.N, self.num_class, self.num_class))
		self.q_x_x_y = np.zeros((self.N-1, self.num_class, self.num_class))
		backward_step = np.zeros((self.N, self.num_class))

		# q(X_n|X_{n-1})
		q_xn_xn_1 = self.A

		# P(X_n|y_1, ...y_N) initialised separately
		backward_step[-1, :] = self.q_xn_given_yn[-1, :]

		# Future conditioning and backward step for all n-{N}
		for row in range(self.N-2, -1, -1):
			y_cap = np.array(y[row+1,:]).reshape((self.D, 1))
			
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
		self.A = np.sum(self.q_x_x_y, axis=0)
		self.A = self.A/np.sum(self.A, axis=0)

		# Updating Sigma
		self.Sigma = [np.cov(Y.T, aweights=self.q_x_y[:,i]) for i in range(self.num_class)]
		self.Sigma_inv = [np.linalg.pinv(S) for S in self.Sigma]
		self.k = [1.0/((((2*math.pi)**self.D)*np.linalg.det(S))**0.5) for S in self.Sigma]

		# Updating C
		self.C = [np.sum(self.q_x_y[:,i].reshape(self.N,1)*Y, axis=0)/(np.sum(self.q_x_y[:,i])) for i in range(self.num_class)]



	def Expectation_Maximization(self, Y):

		# Initialize parameters using data
		self.initialize_params(Y)

		print("\n\n\n", "-"*60, sep="")
		print("\t\tExpectation Maximization")
		print("-"*60, "\n\n")

		print("Running.....")

		self.Filtering(Y)
		self.Smoothing(Y)

		# Iterations
		for e in range(self.epochs):
			self.Maximization(Y)
			self.Filtering(Y)
			self.Smoothing(Y)

			# Stopping Criteria
			err = np.mean((self.q_x_y-self.q_x_y_prev)**2)
			print(f"Epoch {e+1} ====> Err : {err}")
			if err < 3e-5: 
				break 

		print("Finished.....")

		print("\n\n\n", "-"*60, sep="")
		print("\t\tFinal Parameters")
		print("-"*60, "\n\n")
		print("pi : ", self.pi)
		print("\n\nA : \n", self.A)
		print("\n\n\nC : \n", self.C)
		print("\n\n\nSigma : \n", self.Sigma)


		return self.q_x_y, self.Sigma, self.C