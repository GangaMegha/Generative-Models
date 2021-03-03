'''
Program Description :
	Perform ELIMINATE Algorithm on a linear directed graphical model to compute P(X=a|Y=b)

	Given 
		1. Total number of nodes : N
		2. No. of values each variable can take : m
		2. Marginal probabilities at each node P(Xi) - input as a row followed by row
			(Input taken as Probabilities of node 1 taking each of the values followed by node 2 and so on) 
		   	(variables indicated by row i and values indicated by column j, 
		   	index ij contains the probability of that variable taking that value)
		3. Conditional probabilities between every 2 nodes - input as a row followed by a row
			(for a row, index i to column index j => P( X_i+1=val(i)|X_i=val(j) )
			(row index i to column index j => P( X_i+1=val(i)|X_i=val(j))
		4. Required Conditional Probability to be computed P(X=a|Y=b)

	Output 
'''

import argparse
import pandas as pd
import numpy as np

# Function for displaying the provided data
def DisplayValues(args):
	print("\n\n\n\n\n*************************************************")
	print(" Displaying the given data..............")
	print("*************************************************\n\n")

	print("\nMarginal Probabilities : \n")
	Nodes = ["X"+str(i) for i in range(args.nodes)]
	Values = [str(i) for i in range(args.numvals)]
	Marginals = np.array([float(i) for i in args.marginal_p.split(' ')]).reshape((args.nodes, args.numvals))
	Marginals_ = pd.DataFrame(data=Marginals, index=Nodes, columns=Values)
	print(Marginals_)

	print("\nConditional Probabilities P(X_{i+1}|X_i): \n")
	print("\tX_i")
	Conditionals = np.array([float(i) for i in args.conditional_p.split(' ')]).reshape((args.numvals, args.numvals))
	Conditionals_ = pd.DataFrame(data=Conditionals, index=Values, columns=Values)
	print("X_{i+1}")
	print(Conditionals_)

	return(Marginals, Conditionals)


# Function for performing steps in ELIMINATE
def ELIMINATE(Marginals, Conditionals, order, values, X, a, Y, b) :
	print("\n\n\n\n***********************************")
	print("\t Begin ELIMINATE")
	print("***********************************")
	Delta = [1 if i==b else 0 for i in range(len(Marginals[Y-1, :]))]
	print("\n\n\nEVIDENCE............................. \n\n Evidence Potential, Delta_{}: {}".format(Y, Delta))

	print("\n\n\nUPDATE..............................")

	print("\n\n Printing the elements of the Active List computed via ELIMINATE : \n\n1. P(X{})={}".format(X,Marginals[X-1,a]))

	message = np.zeros_like(Marginals)

	count = 1

	for i in order[:-1] :
		if(i==Y) :
			for v in range(values) :
				message[i-1,v] = sum(j*k for (j,k) in zip(Delta, Conditionals[:,v])) # m4(X3=0) = 0*P(X4=0|X3=0) + 1*P(X4=1|X3=0)

		else :
			for v in range(values) :
				message[i-1,v] = sum(j*k for (j,k) in zip(Conditionals[:,v], message[i,:])) # m3(X2=0) = P(X3=0|X2=0)*m4(X3=0) + P(X3=1|X2=0)*m4(X3=1)

		count+=1
		print("\n{}. Marginalizing over node X{} gives\n".format(count, i))
		print(pd.DataFrame(data=[message[i-1,:]], index=["m{}(X{})".format(i, i-1)], columns=["X{}={}".format(i-1, v) for v in range(values)]))
	

	print("\n\n\nNORMALIZE...............................")
	P_Y_b = sum(j*k for (j,k) in zip(Marginals[X-1,:],message[X, :])) # P(X4=1) = P(X1=0|X1=0)*m2(X1=0) + P(X2=1|X1=0)*m2(X1=1)

	P_X_a_Y_b = Marginals[X-1,a]*message[X,a] # P(X=a, Y=b) = P(X=a)*m2(X=a)

	print("\n\n\t P(X{}={}) = {}".format(Y, b, P_Y_b))
	print("\n\t P(X{}={}, X{}={}) = {}".format(X, a, Y, b, P_X_a_Y_b))

	return(P_X_a_Y_b/P_Y_b)


'''
For detailed explanation of the elimination steps for the default example :

	message_4[0] = sum(j*k for (j,k) in zip(Delta, Conditionals[:,0])) # 0*P(X4=0|X3=0) + 1*P(X4=1|X3=0)
	message_4[1] = sum(j*k for (j,k) in zip(Delta, Conditionals[:,1])) # 0*P(X4=0|X3=1) + 1*P(X4=1|X3=1)

	message_3[0] = Conditionals[0,0]*message_4[0] + Conditionals[1,0]*message_4[1] # P(X3=0|X2=0)*m4(X3=0) + P(X3=1|X2=0)*m4(X3=1)
	message_3[1] = Conditionals[0,1]*message_4[0] + Conditionals[1,1]*message_4[1] # P(X3=0|X2=1)*m4(X3=0) + P(X3=1|X2=1)*m4(X3=1)

	message_2[0] = Conditionals[0,0]*message_3[0] + Conditionals[1,0]*message_3[1] # P(X2=0|X1=0)*m3(X2=0) + P(X2=1|X1=0)*m3(X2=1)
	message_2[1] = Conditionals[0,1]*message_3[0] + Conditionals[1,1]*message_3[1] # P(X2=0|X1=1)*m3(X2=0) + P(X2=1|X1=1)*m3(X2=1)

	message_1 = Marginals[X,0]*message_2[0] + Marginals[X,1]*message_2[1] # P(X1=0|X1=0)*m2(X1=0) + P(X2=1|X1=0)*m2(X1=1)

	marginal_XY = Marginals[X,a]*message_2[a] # P(X=a, Y=b) = P(X=a)*m2(X=a)

	return(marginal_XY/message_1)
'''

def main():

	print("\n\n\n ~~~~~~~~~~~~~~~~~~~~~~~~~~START PROGRAM~~~~~~~~~~~~~~~~~~~~~~~~~")

	# Displaying the given data
	Marginals, Conditionals = DisplayValues(args)

	# Eliminate Algorithm
	p = ELIMINATE(Marginals, Conditionals, [int(i) for i in args.order.split(' ')], args.numvals, args.X, args.a, args.Y, args.b)

	print("\n\nHence, \tP(X{}={}|X{}={}) = {} \n\nRounding to 3 decimals -----> P(X{}={}|X{}={}) = {:.3f}".format(args.X, args.a, args.Y, args.b, p, args.X, args.a, args.Y, args.b, p))

	print("\n\n\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END PROGRAM~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n")


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--nodes", 	 dest='nodes', 	 type=int, default=4, 		  action='store', help="Total number of nodes in the linear Directed Graph")
	parser.add_argument("--numvals", dest='numvals', type=int, default=2, 		  action='store', help="Total number of values the random variable can take (Assumes it takes from 0 to numvals-1)")

	parser.add_argument("--order", 	 	   dest='order',   		 type=str, default="4 3 2 1", 						  action='store', help="Order of elimination")
	parser.add_argument("--conditional_p", dest='conditional_p', type=str, default="0.6 0.2 0.4 0.8", 				  action='store', help="Conditional probability of node i+1 taking value of row given node i takes value of column")
	parser.add_argument("--marginal_p",    dest='marginal_p', 	 type=str, default="0.5 0.5 0.6 0.4 0.3 0.7 0.1 0.9", action='store', help="Marginal probabilities for each value of each node")
	
	parser.add_argument("--X", dest='X', type=int, default=1, action='store', help="Value of X in P(X=a|Y=b)")
	parser.add_argument("--a", dest='a', type=int, default=0, action='store', help="Value of a in P(X=a|Y=b)")
	parser.add_argument("--Y", dest='Y', type=int, default=4, action='store', help="Value of Y in P(X=a|Y=b)")
	parser.add_argument("--b", dest='b', type=int, default=1, action='store', help="Value of b in P(X=a|Y=b)")

	args = parser.parse_args()

	main()