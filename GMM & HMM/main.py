import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import math
from scipy.stats import multivariate_normal

from EM_GMM import GMM
from EM_HMM import HMM
from Var_EM_GMM import GMM_VarEM

# Read data from file
def loadData(filename):
	data = pd.read_csv(filename, sep=',', header=None)
	return data

# Plot data as a scatter plot
def plotData(ax, data, label):
	ax.scatter(data[data[label]==0][1], data[data[label]==0][2], marker='o', c='b', label="x=0")
	ax.scatter(data[data[label]==1][1], data[data[label]==1][2], marker='o', c='g', label="x=1")
	ax.scatter(data[data[label]==2][1], data[data[label]==2][2], marker='o', c='r', label="x=2")
	ax.legend()

def create_plot(ax, title):
	ax.set_title(title)
	ax.set_xlabel('y1')
	ax.set_ylabel('y2')

# Plot Gaussian Contours
def	plot_Gaussian(ax, y, sigma, mu):
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

# Compute accuract in %
def Accuracy(true, pred):
	return np.mean(true==pred)*100

def CorrectPredictions(pred, order) :
	pred[:, [0, 1, 2]] = pred[:, order]
	return pred.tolist()


def GMM_EM(config, data, args, X, Y):
	# Create GMM instance
	gmm = GMM(config)

	# Train GMM using Expectation Maximization 
	q_x_given_y, Sigma, C = gmm.Expectation_Maximization(Y)

	data["EM_GMM_Pred"] = CorrectPredictions(q_x_given_y, [2, 1, 0])
	data["EM_GMM_label"] = data["EM_GMM_Pred"].apply(lambda x: np.argmax(x))

	print("\n\n\nAccuracy of GMM:", Accuracy(data[["EM_GMM_label"]].values, data[[args.label_idx]].values))


	#--------------- Plot Gaussians ---------------------
	fig, ax = plt.subplots()
	create_plot(ax, "Predicted Gaussians: GMM with EM")
	plotData(ax, data, args.label_idx)
	for i in range(config["num_class"]):
		plot_Gaussian(ax, Y, Sigma[i], C[i])
	

	#--------------- Create prediction plots ---------------------
	fig, ax = plt.subplots(1, 2)
	create_plot(ax[0], "Data using x")
	plotData(ax[0], data, args.label_idx)

	create_plot(ax[1], "Class prediction: GMM with EM")
	plotData(ax[1], data, "EM_GMM_label")

	plt.show()

	return data


def HMM_EM(config, data, args, X, Y):
	# Create HMM instance
	hmm = HMM(config)

	# Train HMM using Expectation Maximization 
	q_x_given_y, Sigma, C = hmm.Expectation_Maximization(Y)

	data["EM_HMM_Pred"] = CorrectPredictions(q_x_given_y, [2, 1, 0])
	data["EM_HMM_label"] = data["EM_HMM_Pred"].apply(lambda x: np.argmax(x))

	print("\n\n\nAccuracy of HMM:", Accuracy(data[["EM_HMM_label"]].values, data[[args.label_idx]].values))


	#--------------- Plot Gaussians ---------------------
	fig, ax = plt.subplots()
	create_plot(ax, "Predicted Gaussians: HMM with EM")
	plotData(ax, data, args.label_idx)
	for i in range(config["num_class"]):
		plot_Gaussian(ax, Y, Sigma[i], C[i])
	

	#--------------- Create prediction plots ---------------------
	fig, ax = plt.subplots(1, 2)
	create_plot(ax[0], "Data using x")
	plotData(ax[0], data, args.label_idx)

	create_plot(ax[1], "Class prediction: HMM with EM")
	plotData(ax[1], data, "EM_HMM_label")

	plt.show()

	return data

def GMM_Var_EM(config, data, args, X, Y):
	# Create GMM instance
	gmm = GMM_VarEM(config)

	# Train GMM using variational Expectation Maximization 
	q_x_given_y = gmm.Expectation_Maximization(Y)

	data["Var_EM_GMM_Pred"] = CorrectPredictions(q_x_given_y, [0, 2, 1])
	data["Var_EM_GMM_label"] = data["Var_EM_GMM_Pred"].apply(lambda x: np.argmax(x))

	print("\n\n\nAccuracy of HMM:", Accuracy(data[["Var_EM_GMM_label"]].values, data[[args.label_idx]].values))
	

	#--------------- Create prediction plots ---------------------
	fig, ax = plt.subplots(1, 2)
	create_plot(ax[0], "Data using x")
	plotData(ax[0], data, args.label_idx)

	create_plot(ax[1], "Class prediction: Bayesian GMM with Variational EM")
	plotData(ax[1], data, "Var_EM_GMM_label")

	plt.show()

	return data
def main(args):
	config = vars(args)

	# Load Data
	data = loadData(args.data_file)

	# Infer classes
	config["num_class"] = len(data[args.label_idx].unique())

	# Label
	X = data[args.label_idx].values

	# Data
	Y = data.drop(args.label_idx, axis=1).values


	choice = 0
	while(choice!="5") :

		print("\n\n", "="*60, sep="")
		print("\t\t\tMENU")
		print("="*60)
		print("\n\t1. Plot Data")
		print("\t2. EM for Gaussian Mixture Model")
		print("\t3. EM for Hidden Markov Model")
		print("\t4. Variational EM for Bayesian Gaussian Mixture Model")
		print("\t5. Exit")
		print("="*60)

		choice = input("\nEnter your choice : ")

		if(choice=="1"):
			print("\nDisplaying input data : ")
			print(data.head())

			fig, ax = plt.subplots()
			create_plot(ax, "Data")
			plotData(ax, data, args.label_idx)
			plt.show()

		elif(choice=="2"):

			# EM algorithm for Gaussian Mixture Models
			data = GMM_EM(config, data, args, X, Y)

		elif(choice=="3"):

			# EM algorithm for Hidden Markov Models
			data = HMM_EM(config, data, args, X, Y)
			
		elif(choice=="4"):
			
			# Variational EM algorithm for Gaussian Mixture Models
			GMM_Var_EM(config, data, args, X, Y)

		elif(choice=="5"):
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
	parser.add_argument("--label_idx", dest='label_idx', type=int, default=0,  action='store', help="column index of true labels in data (assumed to be integer valued, starting from 0)")
	parser.add_argument("--epochs", dest='epochs', type=int, default=5000,  action='store', help="no. of epochs over the data")
	args = parser.parse_args()
	main(args)