import numpy as np 
import math
from scipy.stats import multivariate_normal
from scipy.special import digamma


class GMM_VarEM():

	def __init__(self, config):

		print("\n\n\n\n\n\n", "*"*60, sep="")
		print("\t\tGaussian Mixture Models\n")
		print("\t    Latent Variable Density Estimation")
		print("*"*60, "\n\n")

		# Updating dictionary
		self.__dict__.update(config)


	# FUnction for initializing parameters
	def initialize_params(self, X):

		print("\n\n\n", "-"*60, sep="")
		print("\t\tInitializing params")
		print("-"*60, "\n\n")

		# Number of observations (n = 1 to N)
		self.N = X.shape[0]

		# Dimention of data
		self.D = X.shape[1]

		# alpha_0 for Dirichlet prior for the mixing coefficients pi
		# initialized to 1 (small => posterior distribution will be influenced primarily by data than by the prior)
		self.alpha_0 = 1

		# m_0 : mean of Gaussian component in Gaussian-Wishart prior for joint probability over mu and lambda
		self.m_0 = np.zeros(self.D)

		# beta_0 : constant term in covariance of Gaussian component in Gaussian-Wishart prior for joint probability over mu and lambda
		self.beta_0 = 1

		# nu_0 : initial no. of degrees of freedom in the Wishart component in Gaussian-Wishart prior for joint probability over mu and lambda
		self.nu_0 = self.D + 1

		# W_0 : initial W_0 in the Wishart component in Gaussian-Wishart prior for joint probability over mu and lambda
		self.W_0 = np.identity(self.D)


		print("N : ", self.N)
		print("\nD : ", self.D)
		print("\nalpha_0 : ", self.alpha_0)
		print("\nm_0 : ", self.m_0)
		print("\nbeta_0 : ", self.beta_0)
		print("\nnu_0 : ", self.nu_0)

		# N_k
		self.N_k = np.zeros(self.num_class)

		# x_bar_k
		self.x_bar_k = [None]*self.num_class

		# S_k
		self.S_k = [None]*self.num_class

		# alpha_k
		self.alpha_k = np.ones(self.num_class)*self.alpha_0

		# beta_k
		self.beta_k = np.ones(self.num_class)*self.beta_0

		# m_k
		self.m_k = [X[n, :] for n in range(self.num_class)]
		# self.m_k = [self.m_0]*self.num_class

		# W_k
		self.W_k = [self.W_0]*self.num_class

		# nu_k
		self.nu_k = np.ones(self.num_class)*self.nu_0

		# E_k = E_mu_lambda[(x_n - mu_k).T lambda ((x_n - mu_k))]
		self.E_k = np.zeros((self.N, self.num_class))

		# lambda_tilda
		self.lambda_tilda = np.zeros(self.num_class)

		# pi_tilda
		self.pi_tilda = np.ones(self.num_class)*1.0/self.num_class

		# Posterior probability
		self.r_n_k = np.zeros((self.N, self.num_class))
		self.r_n_k_prev = np.zeros((self.N, self.num_class))


	# Gaussian Mixture Model Inference			
	def Expectation(self, X):

		# E_k = E_mu_lambda[(x_n - mu_k).T lambda ((x_n - mu_k))]
		for k in range(self.num_class):
			for n in range(self.N):
				self.E_k[n,k] = self.D/self.beta_k[k] + self.nu_k[k] * ( (X[n, :]-self.m_k[k]).T @ self.W_k[k] @ (X[n, :]-self.m_k[k]) )


		# lambda_tilda
		for k in range(self.num_class):
			self.lambda_tilda[k] = 0 

			for i in range(1, self.D+1) :
				self.lambda_tilda[k] += digamma((self.nu_k[k]+1-i)/2)
			
			self.lambda_tilda[k] += self.D*np.log(2) + np.log(np.linalg.det(self.W_k[k]))

		self.lambda_tilda = np.exp(self.lambda_tilda)


		# pi_tilda
		for k in range(self.num_class):
			self.pi_tilda[k] = digamma(self.alpha_k[k]) - digamma(np.sum(self.alpha_k))

		self.pi_tilda = np.exp(self.pi_tilda)


		# Computing posterior probability
		self.r_n_k_prev = self.r_n_k.copy()

		for n, x in enumerate(X):
			for k in range(self.num_class):
				self.r_n_k[n, k] = self.pi_tilda[k] * (self.lambda_tilda[k]**0.5) * np.exp(-0.5*self.E_k[n, k])

			self.r_n_k[n,:] = self.r_n_k[n,:]/(np.sum(self.r_n_k[n,:]))


	def Maximization(self, X):

		# N_k
		for k in range(self.num_class):
			self.N_k[k] = np.sum(self.r_n_k[:, k])

		# x_bar_k
		for k in range(self.num_class):
			self.x_bar_k[k] = (1.0/self.N_k[k])*np.sum(self.r_n_k[:, k].reshape(self.N, 1)*X, axis=0)

		# S_k
		for k in range(self.num_class):
			summ = np.zeros((self.D, self.D))
			for n in range(self.N):
				C = (X[n, :] - self.x_bar_k[k]).reshape(self.D, 1)
				summ += (1.0/self.N_k[k]) * self.r_n_k[n, k] * ( C @ C.T )
			self.S_k[k] = summ

		# alpha_k
		self.alpha_k = self.alpha_0 + self.N_k

		# beta_k
		self.beta_k = self.beta_0 + self.N_k

		# m_k
		self.m_k = np.divide( (self.beta_0*self.m_0 + self.N_k.reshape(self.num_class, 1)*self.x_bar_k), self.beta_k.reshape(self.num_class, 1) )

		# W_k
		for k in range(self.num_class):
			C = (self.x_bar_k[k] - self.m_0).reshape(self.D, 1)
			self.W_k[k] = np.linalg.pinv(self.W_0) + self.N_k[k]*self.S_k[k] + (self.beta_0*self.N_k[k]/(self.beta_0+self.N_k[k]))*(C @ C.T)
			self.W_k[k] = np.linalg.pinv(self.W_k[k])

		# nu_k
		self.nu_k = self.nu_0 + self.N_k


	def Expectation_Maximization(self, X):

		# Initialize parameters using data
		self.initialize_params(X)

		print("\n\n\n", "-"*60, sep="")
		print("\t\tVariational Expectation Maximization")
		print("-"*60, "\n\n")

		print("Running.....")

		self.Expectation(X)

		# Iterations
		for e in range(self.epochs):
			self.Maximization(X)
			self.Expectation(X)

			# Stopping Criteria
			err = np.mean((self.r_n_k-self.r_n_k_prev)**2)
			print(f"Epoch {e+1} ====> Err : {err}")
			if err < 2e-6:
				break 

		print("Finished.....")

		print("\n\n\n", "-"*60, sep="")
		print("\t\tFinal Parameters")
		print("-"*60, "\n\n")

		print("\nalpha_k : ", self.alpha_k)
		print("\nm_k : ", self.m_k)
		print("\nbeta_k : ", self.beta_k)
		print("\nnu_k : ", self.nu_k)

		return self.r_n_k