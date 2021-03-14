import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import math
from scipy.stats import multivariate_normal

def loadData(filename, columns):
	x = np.loadtxt(filename)
	column_names = columns.split(" ")
	data = pd.DataFrame(x, columns = column_names)

	return data

def plotData(ax, data, column_names):
	# fig = plt.figure()
	# fig.add_subplot(111)

	ax.scatter(data[data["y"]==1]["x1"], data[data["y"]==1]["x2"], marker='x', c='r', label="y=1")
	ax.scatter(data[data["y"]==0]["x1"], data[data["y"]==0]["x2"], marker='o', c='g', label="y=0")
	# plt.show()

def plotx1x2(ax, x1, x2, line, label):
	ax.plot(x1, x2, line, label=label)
	ax.legend(loc='lower right')

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


def Gaussian_Class_Conditioned_Density(data, p):

	print("\n\n\n\n\n\n", "*"*60, sep="")
	print("\tFitting Gaussian Class Conditional Density")
	print("*"*60, "\n\n")

	print("Displaying input data : ")
	print(data[data["y"]==1].head())

	pi = len(data[data["y"]==1].index)*1.0/len(data.index)
	print("pi=probability(y=1) calculated from data :", pi)

	mu0 = data[data["y"]==0][["x1", "x2"]].mean(axis=0)
	mu1 = data[data["y"]==1][["x1", "x2"]].mean(axis=0)
	print("mu_0 (Mean of data where y=0) :", mu0)
	print("mu_1 (Mean of data where y=1):", mu1)

	#------------------- Different Sigma ----------------------
	print("\n\nFitting p(x|y_k, mu_k, sigma_k) ........")

	fig, ax = plt.subplots(2,2)
	ax[0,0].set_xlim([min(data["x1"])-2, max(data["x1"])+2])
	ax[0,0].set_ylim([min(data["x2"])-2, max(data["x2"])+2])
	ax[0,0].set_title(r"GMM : $p(x|y_k, \mu_k, \Sigma_k)$ ; Sigma depends on y")
	ax[0,0].set_xlabel('x1')
	ax[0,0].set_ylabel('x2')	
	plotData(ax[0,0], data, data.columns)
	plot_Gaussian(ax[0,0], data[data["y"]==0][["x1", "x2"]], data[data["y"]==0][["x1", "x2"]].cov(), mu0.values)
	plot_Gaussian(ax[0,0], data[data["y"]==1][["x1", "x2"]], data[data["y"]==1][["x1", "x2"]].cov(), mu1.values)


	#------------------- Same Sigma : Independent of y : Cov Matrix ----------------------

	print("Fitting p(x|y_k, mu_k, sigma) .......")

	var = data[["x1", "x2"]].var(axis=0)
	print("Variance of data (independent of class):", var)

	sigma = data[["x1", "x2"]].cov()
	print("Sigma (Covariance metrix) :\n", sigma)
	sigma_inv = pd.DataFrame(np.linalg.pinv(sigma.values), data[["x1", "x2"]].columns, data[["x1", "x2"]].columns)
	print("Sigma inverse :\n", sigma_inv)

	beta = sigma_inv.dot(mu1 - mu0)
	gamma = -0.5*((mu1-mu0).T).dot(sigma_inv).dot(mu1+mu0) + math.log(pi/(1.0-pi))
	print("Beta :", beta)
	print("Gamma :", gamma)

	p_y_1 = 1/(1 + math.e**(-1*(data[["x1", "x2"]]).dot(beta) - gamma))

	x1 = np.linspace(start=min(data[["x1"]].values), stop=max(data[["x1"]].values), num=150)
	x2_1 = 1.0/beta[1]*( -beta[0]*x1 - gamma - math.log(1.0/p - 1))
	print("Fitting p(x|y_k, mu_k, sigma) .......")

	ax[0,1].set_xlim([min(data["x1"])-2, max(data["x1"])+2])
	ax[0,1].set_ylim([min(data["x2"])-2, max(data["x2"])+2])
	ax[0,1].set_title(r"GMM : $p(x|y_k, \mu_k, \Sigma)$ ; Sigma independent of y")
	ax[0,1].set_xlabel('x1')
	ax[0,1].set_ylabel('x2')	
	plotData(ax[0,1], data, data.columns)
	plot_Gaussian(ax[0,1], data[data["y"]==0][["x1", "x2"]], sigma, mu0.values)
	plot_Gaussian(ax[0,1], data[data["y"]==1][["x1", "x2"]], sigma, mu1.values)
	plotx1x2(ax[0,1], x1, x2_1, 'k', 'p = {}'.format(p))

	#------------------- Same Sigma : Independent of y : Diagonal Matrix ----------------------

	print("Fitting p(x|y_k, mu_k, sigma) .......")

	var = data[["x1", "x2"]].var(axis=0)
	print("Variance of data (independent of class):", var)

	sigma = np.diag(var)
	print("Sigma (Covariance metrix with variance on its diagonal) :\n", sigma)
	sigma_inv = pd.DataFrame(np.linalg.pinv(sigma), data[["x1", "x2"]].columns, data[["x1", "x2"]].columns)
	print("Sigma inverse :\n", sigma_inv)

	beta_d = sigma_inv.dot(mu1 - mu0)
	gamma_d = -0.5*((mu1-mu0).T).dot(sigma_inv).dot(mu1+mu0) + math.log(pi/(1.0-pi))
	print("Beta :", beta_d)
	print("Gamma :", gamma_d)

	p_y_2 = 1/(1 + math.e**(-1*(data[["x1", "x2"]]).dot(beta_d) - gamma_d))

	x2_2 = 1.0/beta_d[1]*( -beta_d[0]*x1 - gamma_d - math.log(1.0/p - 1))

	ax[1,0].set_xlim([min(data["x1"])-2, max(data["x1"])+2])
	ax[1,0].set_ylim([min(data["x2"])-2, max(data["x2"])+2])
	ax[1,0].set_title(r"GMM : $p(x|y_k, \mu_k, \Sigma_{diag})$ ; Sigma (Diagonal) independent of y")
	ax[1,0].set_xlabel('x1')
	ax[1,0].set_ylabel('x2')	
	plotData(ax[1,0], data, data.columns)
	plot_Gaussian(ax[1,0], data[data["y"]==0][["x1", "x2"]], sigma, mu0.values)
	plot_Gaussian(ax[1,0], data[data["y"]==1][["x1", "x2"]], sigma, mu1.values)
	plotx1x2(ax[1,0], x1, x2_2, 'k', 'p = {}'.format(p))
	# beta_d = pd.concat([beta]*, ignore_index=True)

	#------------------- Compare Sigma : Independent of y : Diagonal v/s Cov Matrix ----------------------

	p_y_1 = pd.DataFrame(p_y_1.values, data[["y"]].index, data[["y"]].columns)
	p_y_2 = pd.DataFrame(p_y_2.values, data[["y"]].index, data[["y"]].columns)

	p_y_1["index"] = p_y_1["y"].apply(lambda x: 1.0 if x>=0.5 else 0.0)
	p_y_2["index"] = p_y_2["y"].apply(lambda x: 1.0 if x>=0.5 else 0.0)


	MSE_1 = np.square(np.subtract(data["y"].values, p_y_1["index"].values)).mean()
	MSE_2 = np.square(np.subtract(data["y"].values, p_y_2["index"].values)).mean()

	ax[1,1].set_xlim([min(data["x1"])-2, max(data["x1"])+2])
	ax[1,1].set_ylim([min(data["x2"])-2, max(data["x2"])+2])
	ax[1,1].set_title(r"GMM : Compare effect of Sigma; Diagonal v/s Cov Matrix")
	ax[1,1].set_xlabel('x1')
	ax[1,1].set_ylabel('x2')	
	plotData(ax[1,1], data, data.columns)
	plotx1x2(ax[1,1], x1, x2_1, 'b--', 'Sigma_Cov; p={}; MSE={:.2f}'.format(p, MSE_1))
	plotx1x2(ax[1,1], x1, x2_2, 'k', 'Sigma_Diag; p={}; MSE={:.2f}'.format(p, MSE_2))

	return beta, beta_d, gamma, gamma_d

# def utility_reg():


def Linear_Regression(data, p):
	print("\n\n\n\n\n\n", "*"*60, sep="")
	print("\t\tLinear Regression")
	print("*"*60, "\n\n")
	X = data[["x1", "x2"]].values
	y = data[["y"]].values

	bias = np.ones(shape=y.shape)

	X = np.concatenate((bias, X), 1)

	# Closed form solution
	W = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

	print("Learned weights : \n\n", W, sep="")

	# Plotting
	x1 = np.expand_dims(np.linspace(start=min(data[["x1"]].values)/2.0, stop=max(data[["x1"]].values), num=150), axis=1)
	y = np.ones(shape=x1.shape)*p
	b = np.ones(shape=x1.shape)
	x2 = (y - W[0,0]*x1 - W[2,0]*b)/W[1,0]

	fig, ax = plt.subplots()
	ax.set_title("Linear Regression")
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')	
	plotData(ax, data, data.columns)
	plotx1x2(ax, x1, x2, 'k', 'p = {}'.format(p))

	return W


def Logistic_Regression_IRLS(data, p):
	print("\n\n\n\n\n\n", "*"*60, sep="")
	print("\t\tLogistic Regression")
	print("*"*60, "\n\n")
	X = data[["x1", "x2"]].values
	y = data[["y"]].values

	bias = np.ones(shape=y.shape)

	X = np.concatenate((bias, X), 1)

	# Closed form solution
	W = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

	print("Learned weights : \n\n", W, sep="")

	# Plotting
	x1 = np.expand_dims(np.linspace(start=min(data[["x1"]].values)/2.0, stop=max(data[["x1"]].values), num=150), axis=1)
	y = np.ones(shape=x1.shape)*math.log(p/(1-p))
	b = np.ones(shape=x1.shape)
	x2 = (y - W[0,0]*x1 - W[2,0]*b)/W[1,0]

	fig, ax = plt.subplots()
	ax.set_title("Logistic Regression")
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')	
	plotData(ax, data, data.columns)
	plotx1x2(ax, x1, x2, 'k', 'p = {}'.format(p))

	return W

def Testing(args, beta, beta_d, gamma, gamma_d, W_1, W_2):

	p = args.p

	data = loadData(args.test_file, args.columns)

	fig, ax = plt.subplots()
	ax.set_title("Performance Comparison")
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')	
	plotData(ax, data, data.columns)


	X = data[["x1", "x2"]].values
	y = data[["y"]].values
	bias = np.ones(shape=y.shape)
	Xb = np.concatenate((bias, X), 1)



	# Linear Regression
	p_y_1 = Xb.dot(W_1)
	p_y_1 = pd.DataFrame(p_y_1, data[["y"]].index, data[["y"]].columns)

	p_y_1["index"] = p_y_1["y"].apply(lambda x: 1.0 if x>=0.5 else 0.0)

	MSE_Lin_1 = np.square(np.subtract(y, p_y_1["index"].values)).mean()
	MSE_Lin_2 = np.square(np.subtract(y, p_y_1["y"].values)).mean()


	x1 = np.expand_dims(np.linspace(start=min(data[["x1"]].values)/2.0, stop=max(data[["x1"]].values), num=150), axis=1)
	y_cap = np.ones(shape=x1.shape)*p
	b = np.ones(shape=x1.shape)
	x2 = (y_cap - W_1[0,0]*x1 - W_1[2,0]*b)/W_1[1,0]
	plotx1x2(ax, x1, x2, 'k--', r'Linear Regression: p={}, $MSE_1=${:.2f}; $MSE_2=${:.2f}'.format(p, MSE_Lin_1, MSE_Lin_2))




	# Logistic Regression
	p_y_2 = 1.0/(1+np.exp(Xb.dot(W_2)))
	p_y_2 = pd.DataFrame(p_y_2, data[["y"]].index, data[["y"]].columns)

	p_y_2["index"] = p_y_2["y"].apply(lambda x: 1.0 if x>=0.5 else 0.0)

	MSE_Log_1 = np.square(np.subtract(y, p_y_2["index"].values)).mean()
	MSE_Log_2 = np.square(np.subtract(y, p_y_2["y"].values)).mean()


	x1 = np.expand_dims(np.linspace(start=min(data[["x1"]].values)/2.0, stop=max(data[["x1"]].values), num=150), axis=1)
	y_cap = np.ones(shape=x1.shape)*math.log(p/(1-p))
	b = np.ones(shape=x1.shape)
	x2 = (y_cap - W_2[0,0]*x1 - W_2[2,0]*b)/W_2[1,0]
	plotx1x2(ax, x1, x2, 'b:', r'Logistic Regression: p={}, $MSE_1=${:.2f}; $MSE_2=${:.2f}'.format(p, MSE_Log_1, MSE_Log_2))



	# Gaussian Mixture Model
	p_y_1 = 1/(1 + math.e**(-1*(data[["x1", "x2"]]).dot(beta) - gamma))
	p_y_2 = 1/(1 + math.e**(-1*(data[["x1", "x2"]]).dot(beta_d) - gamma_d))

	p_y_1 = pd.DataFrame(p_y_1.values, data[["y"]].index, data[["y"]].columns)
	p_y_2 = pd.DataFrame(p_y_2.values, data[["y"]].index, data[["y"]].columns)

	p_y_1["index"] = p_y_1["y"].apply(lambda x: 1.0 if x>=0.5 else 0.0)
	p_y_2["index"] = p_y_2["y"].apply(lambda x: 1.0 if x>=0.5 else 0.0)


	MSE_Gmm_11 = np.square(np.subtract(data["y"].values, p_y_1["index"].values)).mean()
	MSE_Gmm_12 = np.square(np.subtract(data["y"].values, p_y_1["y"].values)).mean()

	MSE_Gmm_21 = np.square(np.subtract(data["y"].values, p_y_2["index"].values)).mean()
	MSE_Gmm_22 = np.square(np.subtract(data["y"].values, p_y_2["y"].values)).mean()

	x1 = np.linspace(start=min(data[["x1"]].values), stop=max(data[["x1"]].values), num=150)
	x2_1 = 1.0/beta[1]*( -beta[0]*x1 - gamma - math.log(1.0/p - 1))
	plotx1x2(ax, x1, x2, 'm.', r'GMM $\Sigma_c$: p={}, $MSE_1=${:.2f}; $MSE_2=${:.2f}'.format(p, MSE_Gmm_11, MSE_Gmm_12))
	plotx1x2(ax, x1, x2, 'c-', r'GMM $\Sigma_d$: p={}, $MSE_1=${:.2f}; $MSE_2=${:.2f}'.format(p, MSE_Gmm_21, MSE_Gmm_22))


def main(args):

	p = args.p

	data = loadData(args.train_file, args.columns)

	# fig, ax = plt.subplots(2,2)

	fig, ax = plt.subplots()
	plotData(ax, data, data.columns)

	beta, beta_d, gamma, gamma_d = Gaussian_Class_Conditioned_Density(data, 0.5)

	W_1 = Linear_Regression(data, p)
	W_2 = Logistic_Regression_IRLS(data, p)

	Testing(args, beta, beta_d, gamma, gamma_d, W_1, W_2)

	
	plt.show()


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_file", dest='train_file', type=str, default="classification.dat",  action='store', help=".dat file containing the data")
	parser.add_argument("--test_file", dest='test_file', type=str, default="classification.test", action='store', help=".dat file containing the data")
	parser.add_argument("--columns",  dest='columns',  type=str, default="x1 x2 y",            action='store', help="space separated column names (assumes 2D data with 1 label)")
	parser.add_argument("--p",  dest='p',  type=float, default=0.5,            action='store', help="Input required probability threshold")

	args = parser.parse_args()
	main(args)