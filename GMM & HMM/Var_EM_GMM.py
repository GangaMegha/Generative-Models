import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import math
from scipy.stats import multivariate_normal
from scipy.special import digamma
# Read data from file
def loadData(filename):
	data = pd.read_csv(filename, sep=',', header=None)
	return data

# Compute accuract in %
def Accuracy(true, pred):
	return np.mean(true==pred)*100


class GMM_VarEM():

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

		return(np.argmax(self.r_n_k, axis=1))
	
def assign_label(x):
	if x==2:
		return 1
	elif x==1:
		return 2
	else :
		return 0


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
	gmm = GMM_VarEM(config)

	# Train GMM using Expectation Maximization 
	predictions = gmm.Expectation_Maximization(Y)

	print(X)

	print("\n\n\n")
	print(predictions)

	data["predictions"] = predictions
	data["predictions"] = data["predictions"].apply(lambda x: assign_label(x))

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
	parser.add_argument("--epochs", dest='epochs', type=int, default=5000,  action='store', help="no. of epochs over the data")
	args = parser.parse_args()
	main(args)