import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import math
from scipy.stats import multivariate_normal

# Read data from file
def loadData(filename):
	data = pd.read_csv(filename, sep=',', names = ["x", "y1", "y2"])
	return data


# Plot data as a scatter plot
def plotData(ax, data, label):
	ax.scatter(data[data[label]==0]["y1"], data[data[label]==0]["y2"], marker='o', c='b', label="x=0")
	ax.scatter(data[data[label]==1]["y1"], data[data[label]==1]["y2"], marker='o', c='g', label="x=1")
	ax.scatter(data[data[label]==2]["y1"], data[data[label]==2]["y2"], marker='o', c='r', label="x=2")
	ax.legend()

def create_plot(ax, data, title, label):
	ax.set_title(title)
	ax.set_xlabel('y1')
	ax.set_ylabel('y2')
	plotData(ax, data, label)

# Compute accuract in %
def Accuracy(true, pred):
	return np.mean(true==pred)*100


# Gaussian Mixture Model Inference			
def GMM_Inference(y, pi, C, k, Sigma_inv):

	y = np.array(y).reshape((len(y), 1))
	X = [1, 1, 1]
	X = np.diag(X)

	# Computing posterior probability
	q_x_y = np.zeros(len(pi))

	for i in range(len(pi)):
		x = X[:,i].reshape(X.shape[0], 1)
		q_y_x = k*np.exp(-0.5*(y - C @ x).T @ Sigma_inv @ (y - C @ x))
		q_x = pi[i]
		q_x_y[i] = q_y_x * q_x

	return np.round(q_x_y/np.sum(q_x_y), 3)

# Filtering
def HMM_Filtering(y, k, C, pi, Sigma_inv, A):

	time_update = np.zeros((y.shape[0], C.shape[1]))
	measure_update = np.zeros((y.shape[0], C.shape[1]))

	X = [1, 1, 1]
	X = np.diag(X)

	# q(X_n|X_{n-1})
	q_xn_xn_1 = A

	# 1st measurement update
	for i in range(X.shape[1]):
		# q(y_1|x_1)
		x = X[:,i].reshape(X.shape[0], 1)
		y_cap = np.array(y[0,:]).reshape((y.shape[1], 1))
		q_x1 = pi[i]
		q_y1_given_x1 = k*np.exp(-0.5*(y_cap - C @ x).T @ Sigma_inv @ (y_cap - C @ x))

		# q(x_1|y_1)
		measure_update[0, i] = q_x1 * q_y1_given_x1

	measure_update[0, :] = measure_update[0, :]/np.sum(measure_update[0, :], axis=0)

	# time & measurement update for all n-{1}
	for row in range(1, y.shape[0]):
		for col in range(X.shape[1]):
			# q(y_n|x_n)
			x = X[:,col].reshape(X.shape[0], 1)
			y_cap = np.array(y[row,:]).reshape((y.shape[1], 1))
			q_yn_given_xn = k*np.exp(-0.5*(y_cap - C @ x).T @ Sigma_inv @ (y_cap - C @ x))[0,0]

			# Time Update
			for i in range(X.shape[1]):
				time_update[row, col] += measure_update[row-1, i] * q_xn_xn_1[col, i] 

			# Measurement Update
			measure_update[row, col] = time_update[row, col] *q_yn_given_xn

		measure_update[row, :] = measure_update[row, :]/np.sum(measure_update[row, :], axis=0)

	return measure_update


# Smoothing
def HMM_Smoothing(y, k, C, pi, Sigma_inv, A, filter_Distr):

	future_cond = np.zeros((y.shape[0], C.shape[1], C.shape[1]))
	backward_step = np.zeros((y.shape[0], C.shape[1]))

	X = [1, 1, 1]
	X = np.diag(X)

	# q(X_n|X_{n-1})
	q_xn_xn_1 = A

	# P(X_n|y_1, ...y_N) initialised separately
	backward_step[-1, :] = filter_Distr[-1, :]

	# Future conditioning and backward step for all n-{N}
	for row in range(y.shape[0]-2, -1, -1):
		y_cap = np.array(y[row+1,:]).reshape((y.shape[1], 1))
		
		for i in range(X.shape[1]):	
			for col in range(X.shape[1]):
				# Future Conditioning
				future_cond[row, col, i] = filter_Distr[row, col] * q_xn_xn_1[i, col]

			future_cond[row, :, i] = future_cond[row, :, i]/np.sum(future_cond[row, :, i], axis=0)

		for col in range(X.shape[1]):
			for i in range(X.shape[1]):	
				# Backward step
				backward_step[row, col] += backward_step[row+1, i] * future_cond[row, col, i]

	return backward_step


# Alpha-Recursion to compute Filter Distribution
def HMM_alphaRecusrion(y, k, C, pi, Sigma_inv, A):
	
	alpha = np.zeros((y.shape[0], C.shape[1]))

	X = [1, 1, 1]
	X = np.diag(X)

	# q(X_n|X_{n-1})
	q_xn_xn_1 = A

	constant = 1e2

	# alpha 1 initialised separately
	for i in range(X.shape[1]):
		# q(y_1|x_1)
		x = X[:,i].reshape(X.shape[0], 1)
		y_cap = np.array(y[0,:]).reshape((y.shape[1], 1))
		q_x1 = pi[i]
		q_y1_given_x1 = k*np.exp(-0.5*(y_cap - C @ x).T @ Sigma_inv @ (y_cap - C @ x))

		# alpha_1(x_1)
		alpha[0, i] = q_y1_given_x1 * q_x1

	# alpha_n for all n-{1}
	for row in range(1, y.shape[0]):
		for col in range(X.shape[1]):
			# q(y_n|x_n)
			x = X[:,col].reshape(X.shape[0], 1)
			y_cap = np.array(y[row,:]).reshape((y.shape[1], 1))
			q_yn_given_xn = k*np.exp(-0.5*(y_cap - C @ x).T @ Sigma_inv @ (y_cap - C @ x))[0,0]

			for i in range(X.shape[1]):
				# alpha_n(x_n)
				# constant prevents floating point underflow
				alpha[row, col] += alpha[row-1, i] * q_xn_xn_1[col, i] * constant

			alpha[row, col] *= q_yn_given_xn



	return alpha


# Gamma-Recursion to compute Filter Distribution
def HMM_gammaRecusrion(y, k, C, pi, Sigma_inv, A, alpha):

	gamma = np.zeros((y.shape[0], C.shape[1]), dtype=np.longdouble)
	X = [1, 1, 1]
	X = np.diag(X)

	# q(X_n|X_{n-1})
	q_xn_xn_1 = A

	constant = 1e2

	# gamma -1 initialised separately
	gamma[-1, :] = alpha[-1, :]

	# gamma_n for all n-{N}
	for row in range(y.shape[0]-2, -1, -1):
		y_cap = np.array(y[row+1,:]).reshape((y.shape[1], 1))
		
		for col in range(X.shape[1]):	
			for i in range(X.shape[1]):
				# q(y_{n+1}|x_{n+1})
				x = X[:,i].reshape(X.shape[0], 1)
				q_yn1_given_xn1 = k*np.exp(-0.5*(y_cap - C @ x).T @ Sigma_inv @ (y_cap - C @ x))[0,0]	

				# q(x_{n+1}, x_n | y_1,...y_N)
				# constant prevents floating point underflow
				q_xn_1_xn_y = constant*gamma[row+1, i]*alpha[row, col]*q_xn_xn_1[i, col]*q_yn1_given_xn1/alpha[row+1, i]

				# gamma_n(x_n)
				gamma[row, col] += q_xn_1_xn_y

	return gamma

def HMM(args, data):
	#--------------- Computing input values for inference ---------------------
	# pi
	pi = np.array([float(val) for val in args.pi_hmm.split(',')])
	print("\npi :", pi)

	# C
	C = np.array([[float(val) for val in row.split(',')] for row in args.c_hmm.split(';')])
	print("\nC : \n", C)

	# Sigma of y|x
	Sigma = np.array([[float(val) for val in row.split(',')] for row in args.sigma_hmm.split(';')])
	print("\nSigma y|x: \n", Sigma)
	k = 1.0/(2*math.pi*np.linalg.det(Sigma))
	Sigma_inv = np.linalg.pinv(Sigma)

	# A of x_n|x_{n-1}
	A = np.array([[float(val) for val in row.split(',')] for row in args.a_hmm.split(';')])
	print("\nA x_n|x_{n-1}: \n", A)


	#--------------- filtering ---------------------
	filter_Distr = HMM_Filtering(data[["y1", "y2"]].values, k, C, pi, Sigma_inv, A)

	print("\n\n\nAccuracy of Filter Distribution P(X_n|y_1,...y_n) : {}%".format(Accuracy(data[["x"]].values.flatten(), np.argmax(filter_Distr, axis=1))))


	#--------------- smoothing ---------------------
	smoother = HMM_Smoothing(data[["y1", "y2"]].values, k, C, pi, Sigma_inv, A, filter_Distr)
	# print(gamma[:10])
	print("\n\nAccuracy of Smoother Distribution P(X_n|y_1,...y_N) : {}%".format(Accuracy(data[["x"]].values.flatten(), np.argmax(smoother, axis=1))))


	#--------------- alpha recursion ---------------------
	alpha = HMM_alphaRecusrion(data[["y1", "y2"]].values, k, C, pi, Sigma_inv, A)
	# print(alpha[-10:])

	filter_from_alpha = alpha/(np.tile(np.sum(alpha, axis=1)[:, np.newaxis], (1, alpha.shape[1])))

	print("\n\n\nAccuracy of Filter Distribution P(X_n|y_1,...y_n) (from alpha): {}%".format(Accuracy(data[["x"]].values.flatten(), np.argmax(filter_from_alpha, axis=1))))
	

	#--------------- gamma recursion ---------------------
	gamma = HMM_gammaRecusrion(data[["y1", "y2"]].values, k, C, pi, Sigma_inv, A, alpha)
	# print(gamma[:10])
	print("\n\nAccuracy of Smoother Distribution P(X_n|y_1,...y_N) (from gamma): {}%".format(Accuracy(data[["x"]].values.flatten(), np.argmax(gamma, axis=1))))

	return filter_Distr, smoother, alpha, gamma 

def main(args):

	choice = 0
	beta, beta_d, gamma, gamma_d, W_Lin, W_Log = [], [], [], [], [], []
	data = loadData(args.data_file)

	while(choice!="4") :

		print("\n\n", "="*60, sep="")
		print("\t\t\tMENU")
		print("="*60)
		print("\n\t1. Plot Data")
		print("\t2. GMM Inference")
		print("\t3. HMM Inference")
		print("\t4. Exit")
		print("="*60)

		choice = input("\nEnter your choice : ")

		if(choice=="1"):
			print("\nDisplaying input data : ")
			print(data.head())

			fig, ax = plt.subplots()
			create_plot(ax, data, "Data", "x")
			plt.show()

		elif(choice=="2"):
			#--------------- Computing input values for inference ---------------------
			# pi
			pi = np.array([float(val) for val in args.pi_gmm.split(',')])
			print("\npi :", pi)

			# C
			C = np.array([[float(val) for val in row.split(',')] for row in args.c_gmm.split(';')])
			print("\nC : \n", C)

			# Sigma of y|x
			Sigma = np.array([[float(val) for val in row.split(',')] for row in args.sigma_gmm.split(';')])
			print("\nSigma y|x: \n", Sigma)
			k = 1.0/(2*math.pi*np.linalg.det(Sigma))
			Sigma_inv = np.linalg.pinv(Sigma)


			#--------------- GMM Inference ---------------------
			data["GMM_pred"] = data[["y1", "y2"]].apply(lambda x: GMM_Inference(x, pi, C, k, Sigma_inv), axis=1)

			data["GMM_label"] = data["GMM_pred"].apply(lambda x: np.argmax(x))
			# print(data.head(10))
			print("\n\n\nAccuracy of GMM:", Accuracy(data[["GMM_label"]].values, data[["x"]].values))


			#--------------- Create plots ---------------------
			fig, ax = plt.subplots(1, 2)
			create_plot(ax[0], data, "Data using x", "x")
			create_plot(ax[1], data, "Data class prediction using GMM", "GMM_label")
			plt.show()

		elif(choice=="3"):
			filter_distr, smoother, alpha, gamma = HMM(args, data)

			data["filter_distr"] = filter_distr.tolist()
			data["smoother"] = smoother.tolist()
			data["alpha"] = alpha.tolist()
			data["gamma"] = gamma.tolist()

			data["filter_label"] = data["filter_distr"].apply(lambda x: np.argmax(x))
			data["smoother_label"] = data["smoother"].apply(lambda x: np.argmax(x))
			data["alpha_label"] = data["alpha"].apply(lambda x: np.argmax(x))
			data["gamma_label"] = data["gamma"].apply(lambda x: np.argmax(x))

			# Create plots
			fig, ax = plt.subplots(2, 2)
			create_plot(ax[0, 0], data, r"Data class prediction using Filtering $P(X_n|y_1,...y_n)$", "filter_label")
			create_plot(ax[0, 1], data, r"Data class prediction using Smoothing $P(X_n|y_1,...y_n)$", "smoother_label")
			create_plot(ax[1, 0], data, r"Data class prediction using $\alpha_n(X_n) = P(X_n,y_1,...y_n)$", "alpha_label")
			create_plot(ax[1, 1], data, r"Data class prediction using $\gamma_n(X_n) = P(X_n|y_1,...y_N)$", "gamma_label")
			plt.show()

		elif(choice=="4"):
			data.to_csv(args.out_file, sep="\t")
			print("\n\n Outputs saved in {}".format(args.out_file))
			print("\n\n\t\tEnd of program")

		else :
			print("Invalid option!")
			print("Please choose from options 1-6")
	



if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", dest='data_file', type=str, default="HMMdata.csv",  action='store', help="csv file containing the data")
	parser.add_argument("--out_file", dest='out_file', type=str, default="Output.tsv",  action='store', help="tsv file containing the output predictions")
	parser.add_argument("--pi_gmm", dest='pi_gmm', type=str, default="0.1875, 0.3125, 0.5",  action='store', help="pi for GMM (probability for categorical distribution for X) \nExample: \n\t(3/16, 5/16, 8/16)")
	parser.add_argument("--c_gmm", dest='c_gmm', type=str, default="-2, 2, -2; 0, 2, 2",  action='store', help="C for GMM (Transformation matrix for mean of Y|X) \nExample: \n\t[[-2, 2, -2]\n\t [ 0, 2,  2]]")
	parser.add_argument("--sigma_gmm", dest='sigma_gmm', type=str, default="1, -0.2; -0.2, 2",  action='store', help="sigma of Y|X for GMM (Covariance matrix of Y|X) \nExample: \n\t[[1, -0.2]\n\t [-0.2, 2]]")
	parser.add_argument("--pi_hmm", dest='pi_hmm', type=str, default="0.6, 0.2, 0.2",  action='store', help="pi for HMM (probability for categorical distribution for X_1) \nExample: \n\t(3/5, 1/5, 1/5)")
	parser.add_argument("--c_hmm", dest='c_hmm', type=str, default="-2, 2, -2; 0, 2, 2",  action='store', help="C for HMM (Transformation matrix for mean of Y|X) \nExample: \n\t[[-2, 2, -2]\n\t [ 0, 2,  2]]")
	parser.add_argument("--sigma_hmm", dest='sigma_hmm', type=str, default="1, -0.2; -0.2, 2",  action='store', help="sigma of Y|X for HMM (Covariance matrix of Y|X) \nExample: \n\t[[1, -0.2]\n\t [-0.2, 2]]")
	parser.add_argument("--a_hmm", dest='a_hmm', type=str, default="0.4, 0.2, 0.1; 0.3, 0.5, 0.2; 0.3, 0.3, 0.7",  action='store', help="A of X_n|X_{n-1} for HMM (Transition matrix of Xn|Xn-1) \nExample: \n\t[[0.4, 0.2, 0.1]\n\t [0.3, 0.5, 0.2]\n\t [0.3, 0.3, 0.7]]")
	args = parser.parse_args()
	main(args)