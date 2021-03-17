import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import math
from scipy.stats import multivariate_normal

# Read data from file
def loadData(filename):
	x = np.loadtxt(filename)
	data = pd.DataFrame(x, columns = ["x1", "x2", "y"])

	return data

# Mean squared error between y and y_cap (prediction)
def MSE(y, y_cap):
	return (1.0/len(y))*np.sum((y - y_cap)**2)

def Accuracy(y, y_cap):
	return 1.0*np.sum(y*y_cap + (-y+1)*(-y_cap+1))/len(y)

def Recall(y, y_cap):
	return 1.0*np.sum(y*y_cap)/np.sum(y)

def Precision(y, y_cap):
	return 1.0*np.sum(y*y_cap)/np.sum(y_cap)

#Cross entropy
def cross_entropy(y, y_cap):
	return (-1.0/len(y))*np.sum(y.T * np.log(y_cap))

# Plot data as a scatter plot
def plotData(ax, data, column_names):
	ax.scatter(data[data["y"]==1]["x1"], data[data["y"]==1]["x2"], marker='x', c='r', label="y=1")
	ax.scatter(data[data["y"]==0]["x1"], data[data["y"]==0]["x2"], marker='o', c='g', label="y=0")
	ax.legend(loc='lower right')

# Plot line
def plotx1x2(ax, x1, x2, line, label):
	ax.plot(x1, x2, line, label=label)
	ax.legend(loc='lower right')

# Plot Gaussian Contours
def	plot_Gaussian(ax, data_y_k, sigma, mu):
	x1 = np.linspace(start=min(data_y_k[["x1"]].values), stop=max(data_y_k[["x1"]].values), num=150)
	x2 = np.linspace(start=min(data_y_k[["x2"]].values), stop=max(data_y_k[["x2"]].values), num=150)
	X,Y = np.meshgrid(x1, x2)

	# Probability Density
	rv = multivariate_normal(mu, sigma)
	pos = np.empty(X.shape + (2,))
	pos[:, :, 0] = X
	pos[:, :, 1] = Y
	pdf = rv.pdf(pos)

	ax.contour(X, Y, pdf)

# Set the figure attributes for plotting Gaussian contours
def CreateFig_Gaussian(ax, i, j, data, mu0, mu1, sigma0, sigma1, title):
	ax[i,j].set_xlim([min(data["x1"])-2, max(data["x1"])+2])
	ax[i,j].set_ylim([min(data["x2"])-2, max(data["x2"])+2])
	ax[i,j].set_title(title)
	ax[i,j].set_xlabel('x1')
	ax[i,j].set_ylabel('x2')	
	plotData(ax[i,j], data, data.columns)
	plot_Gaussian(ax[i,j], data[data["y"]==0][["x1", "x2"]], sigma0, mu0.values)
	plot_Gaussian(ax[i,j], data[data["y"]==1][["x1", "x2"]], sigma1, mu1.values)

	
# Compute the posterior probability
# Compute line for given p
def Compute_Gaussian_p(data, sigma, mu0, mu1, pi, x1, p):
	sigma_inv = pd.DataFrame(np.linalg.pinv(sigma), data[["x1", "x2"]].columns, data[["x1", "x2"]].columns)
	print("\nSigma inverse :\n", sigma_inv)

	beta = sigma_inv @ (mu1 - mu0)
	gamma = -0.5 * (mu1-mu0).T @ sigma_inv @ (mu1 + mu0) + math.log(pi/(1.0-pi))
	print("\nBeta :\n", beta, sep="")
	print("\nGamma : ", gamma, sep="")

	p_y = 1.0/(1 + math.e**(-1.0 * data[["x1", "x2"]] @ beta - gamma))

	x2 = 1.0/beta[1]*( -1* beta[0]*x1 - gamma + math.log(p/(1.0-p)) )

	return beta, gamma, p_y, x2

# Fit Gaussian Class COnditioned Density on the data
# Compute the posterior probability
def Gaussian_Class_Conditioned_Density(data, p):

	print("\n\n\n\n\n\n", "*"*60, sep="")
	print("\tFitting Gaussian Class Conditional Density")
	print("*"*60, "\n\n")

	pi = len(data[data["y"]==1].index)*1.0/len(data.index)
	print("\npi = probability(y=1) :", pi)
	print("Note : calculated from data as data(y==1)/data")

	mu0 = data[data["y"]==0][["x1", "x2"]].mean(axis=0)
	mu1 = data[data["y"]==1][["x1", "x2"]].mean(axis=0)
	print("\nmu_0 (Mean of data with y=0) :\n", mu0, sep="")
	print("\nmu_1 (Mean of data with y=1) :\n", mu1, sep="")



	#------------------- Different Sigma ----------------------
	print("\n\n\nFitting p(x|y_k, mu_k, sigma_k) .............\n")

	sigma0 = data[data["y"]==0][["x1", "x2"]].cov()
	sigma1 = data[data["y"]==1][["x1", "x2"]].cov()
	print("\nSigma_0 (Sigma of data with y=0) :\n", sigma0, sep="")
	print("\nSigma_1 (Sigma of data with y=1) :\n", sigma1, sep="")

	fig, ax = plt.subplots(2,2)
	CreateFig_Gaussian(ax, 0, 0, data, mu0, mu1, sigma0, sigma1, r"GMM : $p(x|y_k, \mu_k, \Sigma_k)$ ; Sigma depends on y")



	#------------------- Same Sigma : Independent of y : Cov Matrix ----------------------

	print("\n\n\nFitting p(x|y_k, mu_k, sigma_cov) ............\n")

	sigma_cov = data[["x1", "x2"]].cov().values
	print("\nSigma (Covariance matrix) :\n", sigma_cov)
	
	x1 = np.linspace(start=min(data[["x1"]].values), stop=max(data[["x1"]].values), num=150)

	beta_cov, gamma_cov, p_y_cov, x2_cov = Compute_Gaussian_p(data, sigma_cov, mu0, mu1, pi, x1, p)

	CreateFig_Gaussian(ax, 0, 1, data, mu0, mu1, sigma_cov, sigma_cov, r"GMM : $p(x|y_k, \mu_k, \Sigma_{cov})$ ; Sigma independent of y")
	plotx1x2(ax[0,1], x1, x2_cov, 'b--', 'p = {}'.format(p))



	#------------------- Same Sigma : Independent of y : Diagonal Matrix ----------------------

	print("\n\n\nFitting p(x|y_k, mu_k, sigma) ............\n")

	var = data[["x1", "x2"]].var(axis=0)
	print("\nVariance of data (independent of class):\n", var, sep="")

	sigma_diag = np.diag(var)
	print("\nSigma (Diagonal matrix with variance on its diagonal) :\n", sigma_diag)

	beta_diag, gamma_diag, p_y_diag, x2_diag = Compute_Gaussian_p(data, sigma_diag, mu0, mu1, pi, x1, p)

	CreateFig_Gaussian(ax, 1, 0, data, mu0, mu1, sigma_diag, sigma_diag, r"GMM : $p(x|y_k, \mu_k, \Sigma_{diag})$ ; Sigma (Diagonal) independent of y")
	plotx1x2(ax[1,0], x1, x2_diag, 'k', 'p = {}'.format(p))



	#------------------- Compare Sigma : Independent of y : Diagonal v/s Cov Matrix ----------------------

	p_y_cov = pd.DataFrame(p_y_cov.values, data[["y"]].index, data[["y"]].columns)
	p_y_diag = pd.DataFrame(p_y_diag.values, data[["y"]].index, data[["y"]].columns)

	p_y_cov["index"] = p_y_cov["y"].apply(lambda x: 1.0 if x>=p else 0.0)
	p_y_diag["index"] = p_y_diag["y"].apply(lambda x: 1.0 if x>=p else 0.0)

	ax[1,1].set_xlim([min(data["x1"])-2, max(data["x1"])+2])
	ax[1,1].set_ylim([min(data["x2"])-2, max(data["x2"])+2])
	ax[1,1].set_title(r"GMM : Compare effect of Sigma; Diagonal v/s Cov Matrix")
	ax[1,1].set_xlabel('x1')
	ax[1,1].set_ylabel('x2')	
	plotData(ax[1,1], data, data.columns)
	plotx1x2(ax[1,1], x1, x2_cov, 'b--', 'Sigma_Cov; p={}; MSE={:.2f}'.format(p, MSE(data["y"].values, p_y_cov["y"].values)) )
	plotx1x2(ax[1,1], x1, x2_diag, 'k', 'Sigma_Diag; p={}; MSE={:.2f}'.format(p, MSE(data["y"].values, p_y_diag["y"].values)) )

	return beta_cov, beta_diag, gamma_cov, gamma_diag


# Linear Regression : Closed form solution
# Using normal equations
def Linear_Regression(data, p):
	print("\n\n\n\n\n\n", "*"*60, sep="")
	print("\t\tLinear Regression")
	print("*"*60, "\n\n")

	X = data[["x1", "x2"]].values
	y = 1.0*data[["y"]].values

	bias = np.ones(shape=y.shape)

	X = np.concatenate((bias, X), 1)

	# Closed form solution
	W = np.linalg.pinv(X.T @ X) @ X.T @ y

	y_cap = X @ W

	print("\nLearned weights using closed form solution (Normal Eq), \n\n W : ", W, sep="")

	# Plotting
	x1 = np.linspace(start=min(data[["x1"]].values)/2.0, stop=max(data[["x1"]].values), num=150)
	y = np.ones(shape=x1.shape)*p
	b = np.ones(shape=x1.shape)
	x2 = (y - W[0,0]*x1 - W[2,0]*b)/W[1,0]

	fig, ax = plt.subplots()
	ax.set_title("Linear Regression")
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')	
	plotData(ax, data, data.columns)
	plotx1x2(ax, x1, x2, 'k', 'p={}; MSE={:.2f}'.format(p, MSE(data["y"].values, y_cap)))

	return W

def sigmoid(W, X):
	return 1.0/(1+np.exp(-1*X @ W))

def IRLS(X, y, epochs):

	# Randomly initialize W
	np.random.seed(1234)
	input_size = X.shape[1]
	W = np.random.uniform(low=-1.0/np.sqrt(input_size), high=1.0/np.sqrt(input_size), size=([input_size, 1]))

	# output prediction for present W
	y_cap = sigmoid(W, X)

	for i in range(epochs):

		# Diagonal matrix of y(1-y)
		R = y_cap * (1-y_cap)
		R = np.diag(R.flatten())

		# Update weights using Newton-Raphson update formula 
		Z = np.linalg.pinv(X.T @ R @ X)
		W = W - Z @ X.T @ (y_cap-y)

		# output prediction for present W
		y_cap = sigmoid(W, X)

		print("Loss at epoch {} -----> Cross entropy : {}; MSE : {}".format(i, cross_entropy(y, y_cap), MSE(y, y_cap)))

	return W


def Logistic_Regression_IRLS(data, p, epochs):
	print("\n\n\n\n\n\n", "*"*60, sep="")
	print("\t\tLogistic Regression")
	print("*"*60, "\n\n")
	X = data[["x1", "x2"]].values
	y = data[["y"]].values

	bias = np.ones(shape=y.shape)

	X = np.concatenate((bias, X), 1)

	W = IRLS(X, y, epochs)
	y_cap = sigmoid(W, X)

	print("Learned weights : \n\n", W, sep="")

	# Plotting
	x1 = np.linspace(start=min(data[["x1"]].values)/2.0, stop=max(data[["x1"]].values), num=150)
	y = np.ones(shape=x1.shape)*math.log(p/(1-p))
	b = np.ones(shape=x1.shape)
	x2 = (y - W[0,0]*x1 - W[2,0]*b)/W[1,0]

	fig, ax = plt.subplots()
	ax.set_title("Logistic Regression")
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')	
	plotData(ax, data, data.columns)
	plotx1x2(ax, x1, x2, 'k', 'p={}; MSE={:.2f}'.format(p, MSE(data["y"].values, y_cap)))

	return W

def Testing(args, beta_cov, beta_diag, gamma_cov, gamma_diag, W_Lin, W_Log, p):

	data = loadData(args.test_file)

	fig, ax = plt.subplots()
	ax.set_title("Performance Comparison on Test Data")
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')	
	plotData(ax, data, data.columns)

	# Formatting the data into required form
	X = data[["x1", "x2"]].values
	y = data[["y"]].values
	bias = np.ones(shape=y.shape)
	Xb = np.concatenate((bias, X), 1)


	# Linear Regression
	p_y_Lin = Xb @ W_Lin
	p_y_Lin = pd.DataFrame(p_y_Lin, data[["y"]].index, data[["y"]].columns)

	p_y_Lin["index"] = p_y_Lin["y"].apply(lambda x: 1.0 if x>=p else 0.0)

	x1 = np.linspace(start=min(data[["x1"]].values), stop=max(data[["x1"]].values), num=150)
	y_cap_Lin = np.ones(shape=x1.shape)*p
	b = np.ones(shape=x1.shape)
	x2_Lin = (y_cap_Lin - W_Lin[0,0]*x1 - W_Lin[2,0]*b)/W_Lin[1,0]
	plotx1x2(ax, x1, x2_Lin, 'k--', 'Linear Regression: p={}, MSE={:.2f}; Acc={:.2f}; P={:.2f}; R={:.2f}'.format(p, MSE(y, p_y_Lin["y"].values), Accuracy(y, p_y_Lin["index"].values), Precision(y, p_y_Lin["index"].values), Recall(y, p_y_Lin["index"].values)))



	# Logistic Regression
	p_y_Log = 1.0/(1+np.exp(Xb.dot(W_Log)))
	p_y_Log = pd.DataFrame(p_y_Log, data[["y"]].index, data[["y"]].columns)

	p_y_Log["index"] = p_y_Log["y"].apply(lambda x: 1.0 if x>=0.5 else 0.0)

	MSE_Log_1 = np.square(np.subtract(y, p_y_Log["index"].values)).mean()
	MSE_Log_2 = np.square(np.subtract(y, p_y_Log["y"].values)).mean()


	y_cap_Log = np.ones(shape=x1.shape)*math.log(p/(1-p))
	b = np.ones(shape=x1.shape)
	x2_Log = (y_cap_Log - W_Log[0,0]*x1 - W_Log[2,0]*b)/W_Log[1,0]
	plotx1x2(ax, x1, x2_Log, 'b:', 'Logistic Regression: p={}, MSE={:.2f}; Acc={:.2f}; P={:.2f}; R={:.2f}'.format(p, MSE(y, p_y_Log["y"].values), Accuracy(y, p_y_Log["index"].values), Precision(y, p_y_Log["index"].values), Recall(y, p_y_Log["index"].values)))



	# Gaussian Mixture Model
	p_y_GMM_cov = 1/(1 + math.e**(-1*(data[["x1", "x2"]]).dot(beta_cov) - gamma_cov))
	p_y_GMM_diag = 1/(1 + math.e**(-1*(data[["x1", "x2"]]).dot(beta_diag) - gamma_diag))

	p_y_GMM_cov = pd.DataFrame(p_y_GMM_cov.values, data[["y"]].index, data[["y"]].columns)
	p_y_GMM_diag = pd.DataFrame(p_y_GMM_diag.values, data[["y"]].index, data[["y"]].columns)

	p_y_GMM_cov["index"] = p_y_GMM_cov["y"].apply(lambda x: 1.0 if x>=p else 0.0)
	p_y_GMM_diag["index"] = p_y_GMM_diag["y"].apply(lambda x: 1.0 if x>=p else 0.0)

	x2_GMM_cov  = 1.0/beta_cov[1] *( -1* beta_cov[0]*x1  - gamma_cov + math.log(p/(1.0-p))  )
	x2_GMM_diag = 1.0/beta_diag[1]*( -1* beta_diag[0]*x1 - gamma_diag + math.log(p/(1.0-p)) )
	plotx1x2(ax, x1, x2_GMM_cov, 'm.', 'GMM Sigma_c: p={}, MSE={:.2f}; Acc={:.2f}; P={:.2f}; R={:.2f}'.format(p, MSE(y, p_y_GMM_cov["y"].values), Accuracy(y, p_y_GMM_cov["index"].values), Precision(y, p_y_GMM_cov["index"].values), Recall(y, p_y_GMM_cov["index"].values)))
	plotx1x2(ax, x1, x2_GMM_diag, 'c-', 'GMM Sigma_d: p={}, MSE={:.2f}; Acc={:.2f}; P={:.2f}; R={:.2f}'.format(p, MSE(y, p_y_GMM_diag["y"].values), Accuracy(y, p_y_GMM_diag["index"].values), Precision(y, p_y_GMM_diag["index"].values), Recall(y, p_y_GMM_diag["index"].values)))


def main(args):

	choice = 0
	beta, beta_d, gamma, gamma_d, W_Lin, W_Log = [], [], [], [], [], []
	data = loadData(args.train_file)

	while(choice!=6) :

		print("\n\n", "="*60, sep="")
		print("\t\t\tMENU")
		print("="*60)
		print("\n\t1. Plot Data")
		print("\t2. Gaussian class-conditional densities")
		print("\t3. Logistic Regression (IRLS)")
		print("\t4. Linear Regression (Closed form - Normal Eq)")
		print("\t5. Compare Results on Test set")
		print("\t6. Exit")
		print("="*60)

		choice = int(input("\nEnter your choice : "))

		if(choice in [2, 3, 4, 5]):
			# Threshold probability to plot line
			p = float(input("Enter value of p : "))
			if(p<0 or p>1):
				print("Please enter a value of p in (0,1)")
				continue

		if(choice==1):
			print("\nDisplaying input data : ")
			print(data.head())

			fig, ax = plt.subplots()
			ax.set_title("Train Data")
			ax.set_xlabel('x1')
			ax.set_ylabel('x2')
			plotData(ax, data, data.columns)
			plt.show()

		elif(choice==2):
			beta_cov, beta_diag, gamma_cov, gamma_diag = Gaussian_Class_Conditioned_Density(data, p)
			plt.show()

		elif(choice==3):
			epochs = int(input("Enter the no. of epochs : "))
			W_Log = Logistic_Regression_IRLS(data, p, epochs)
			plt.show()

		elif(choice==4):
			W_Lin = Linear_Regression(data, p)
			plt.show()

		elif(choice==5):
			try :
				Testing(args, beta_cov, beta_diag, gamma_cov, gamma_diag, W_Lin, W_Log, p)
				plt.show()
			except Exception as e:
				print("Exception caught : ", e)
				print("\nPlease train the models first before testing")
				print("Please run options 2, 3 and 4 first")

		elif(choice==6):
			print("\n\n\t\tEnd of program")

		else :
			print("Invalid option!")
			print("Please choose from options 1-6")
	



if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_file", dest='train_file', type=str, default="classification.dat",  action='store', help=".dat file containing the data")
	parser.add_argument("--test_file", dest='test_file', type=str, default="classification.test", action='store', help=".dat file containing the data")
	args = parser.parse_args()
	main(args)