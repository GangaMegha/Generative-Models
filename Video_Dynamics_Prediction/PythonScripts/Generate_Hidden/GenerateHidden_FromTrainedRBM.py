import pickle
import numpy as np

in_dir = "C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\Generate_Hidden\\"
out_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\Generate_Hidden\\"


# Read frames
train_frames = pickle.load( open(in_dir +'\\train_frames.pkl' , 'rb' ))
test_frames = pickle.load( open(in_dir +'\\test_frames.pkl' , 'rb' ))

# Read the rbm learned weights
rbm_dir = "C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\RBM\\"
W, b_h, b_v = pickle.load( open(rbm_dir+'\\weights.pkl' , 'rb' ))

print("Loaded learned weights from RBM")
print("W", W.shape)
print("b_h", b_h.shape)
print("b_v", b_v.shape)

def sigmoid(x):  
	#Sigmoid activation 
	#Implemented interms  of tanh for increased stability
	return .5 * (1 + np.tanh(.5 * x))

def bernoulli_array(prob_array, dim):
	# Simulating Bernoulli from uniform
	sample = np.zeros(dim)

	# Draw x~Uni[0,1]
	uni_sample = np.random.uniform(0, 1, dim)

	# return 1 if x < p else return 0
	diff = uni_sample - prob_array
	coords = np.argwhere(diff<0)
	sample[[*coords.T]] = 1  

	return sample

# ------------------------ Train Data ----------------------------------------

for count in range(5):
	hidden = []
	for i in range(train_frames.shape[0]):

		v = train_frames[i].T

		# Getting hidden states of RBM using frames
		# (h x v) @ (v x b) + (h x 1) = (h x b)
		p_h_v = sigmoid(W @ v + b_h)
		h = bernoulli_array(p_h_v, (p_h_v.shape[0], p_h_v.shape[1]))

		hidden.append(h.T)

	hidden = np.array(hidden)
	print("Train Hidden h ", count, ": ", hidden.shape)

	pickle.dump(hidden, open(f"{out_dir}\\train_h_{count}.pkl" , 'wb' ) )


hidden = []
for i in range(train_frames.shape[0]):

	v = train_frames[i].T

	# Getting hidden states of RBM using frames
	# (h x v) @ (v x b) + (h x 1) = (h x b)
	p_h_v = sigmoid(W @ v + b_h)

	hidden.append(p_h_v.T)

hidden = np.array(hidden)
print("Train Hidden p_h_v : ", hidden.shape)

pickle.dump(hidden, open(f"{out_dir}\\train_p_h_v.pkl" , 'wb' ) )


# ------------------------ Test Data ----------------------------------------

for count in range(5):
	hidden = []
	for i in range(test_frames.shape[0]):

		v = test_frames[i].T

		# Getting hidden states of RBM using frames
		# (h x v) @ (v x b) + (h x 1) = (h x b)
		p_h_v = sigmoid(W @ v + b_h)
		h = bernoulli_array(p_h_v, (p_h_v.shape[0], p_h_v.shape[1]))

		hidden.append(h.T)

	hidden = np.array(hidden)
	print("Test Latent Dynamics h ", count, ": ", hidden.shape)

	pickle.dump(hidden, open(f"{out_dir}\\test_h_{count}.pkl" , 'wb' ) )



hidden = []
for i in range(test_frames.shape[0]):

	v = test_frames[i].T

	# Getting hidden states of RBM using frames
	# (h x v) @ (v x b) + (h x 1) = (h x b)
	p_h_v = sigmoid(W @ v + b_h)

	hidden.append(p_h_v.T)

hidden = np.array(hidden)
print("Test Hidden p_h_v : ", hidden.shape)

pickle.dump(hidden, open(f"{out_dir}\\test_p_h_v.pkl" , 'wb' ) )