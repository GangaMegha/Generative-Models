import numpy as np 
import pickle 

import torch

import imageio

import argparse

import os


#---------------------------- Read Models ------------------------------------
def read_models(args):
	#=========== RBM ==========
	W, b_h, b_v = pickle.load( open(args.model_rbm , 'rb' ))

	#=========== Dynamics ==========
	Dynamics = torch.load(args.model_dynamics)

	#=========== Hidden ==========
	Hidden = torch.load(args.model_hidden)

	return W, b_h, b_v, Dynamics, Hidden


#---------------------------- Read data ------------------------------------
def read_data(args):

	data = pickle.load( open(args.data , 'rb' ))
	z = pickle.load( open(args.z , 'rb' ))

	return data, z[:, 0, :]


#---------------------------- Save video frames as gif ------------------------------------
def make_gif(data, out_dir, filename):
	imageio.mimsave(os.path.join(out_dir, filename), data, fps=24)


#---------------------------- RBM utilities ------------------------------------
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

def Get_rbm_hidden_states(data, W, b_h):
	v = data.T

	# Getting hidden states of RBM using frames
	# (h x v) @ (v x b) + (h x 1) = (h x b)
	p_h_v = sigmoid(W @ v + b_h)
	h = bernoulli_array(p_h_v, (p_h_v.shape[0], p_h_v.shape[1]))

	return h.T

def Get_rbm_visible_states(h_cap, W, b_v):
	h = h_cap.T

	# Getting visible states of RBM using hidden states
	# (v x h) @ (h x b) + (v x 1) = (v x b)
	p_v_h = sigmoid(W.T @ h + b_v)
	v = bernoulli_array(p_v_h, (p_v_h.shape[0], p_v_h.shape[1]))

	return v.T

#---------------------------- State Dynamics ------------------------------------

def smooth_and_sample(z):

	# Sampling step
	noise = np.random.normal(size=z.shape)
	z = z + 0.1*noise/10

	print(np.max(0.1*noise/10))

	# Smoothening step
	new_z = []
	for i in range(z.shape[0]-1):
		new_z.append((z[i] + z[i+1])/2)
	return np.array(new_z)

def sample_step(z):

	# Sampling step
	noise = np.random.normal(size=z.shape)
	z = z + 0.1*noise/10

	return z

def Get_latent_dynamics(Dynamics, hidden, z_init):

	flag = True
	z_size = len(z_init)
	img_size = hidden.shape[-1]

	prev_z = torch.tensor(z_init.reshape((1, 1, z_size)), dtype=torch.float32)

	z = []
	z.append(prev_z.cpu().detach().numpy())

	for j in range(1, hidden.shape[0]):

		X = hidden[j].reshape((1, 1, img_size))

		# Step 2. Prepare inputs
		input_video = torch.tensor(X, dtype=torch.float32)

		# Step 3. Run our forward pass.
		out = Dynamics(input_video, prev_z, flag)

		prev_z = out.clone()
		z.append(prev_z.cpu().detach().numpy())

		flag=False

	return np.array(z).reshape((hidden.shape[0], z_size))


#---------------------------- Generate hidden ------------------------------------
def Get_hidden_from_dynamics(Hidden, z, h_init):

	flag = True
	h_size = len(h_init)
	z_size = z.shape[-1]

	prev_h = torch.tensor(h_init.reshape((1, 1, h_size)), dtype=torch.float32)

	h = []
	h.append(prev_h.cpu().detach().numpy())

	for j in range(1, z.shape[0]):

		X = z[j].reshape((1, 1, z_size))

		# Step 2. Prepare inputs
		input_states = torch.tensor(X, dtype=torch.float32)

		# Step 3. Run our forward pass.
		out = Hidden(input_states, prev_h, flag)

		prev_h = out.clone()
		h.append(prev_h.cpu().detach().numpy())

		flag=False

	return np.array(h).reshape((z.shape[0], h_size))


def Generate_video(data, z_init, W, b_h, b_v, Dynamics, Hidden, args, filename, sample=True, smooth=True):

	hidden = Get_rbm_hidden_states(data, W, b_h)

	z_cap = Get_latent_dynamics(Dynamics, hidden, z_init)

	if(sample and smooth):
		z_cap = smooth_and_sample(z_cap)
	elif(sample):
		z_cap = sample_step(z_cap)

	h_cap = Get_hidden_from_dynamics(Hidden, z_cap, hidden[0])

	v_cap = Get_rbm_visible_states(h_cap, W, b_v)

	make_gif(v_cap.reshape((v_cap.shape[0], args.img_height, args.img_width)), args.out_dir, filename)


def main(args):
	
	# Read the trained models
	W, b_h, b_v, Dynamics, Hidden = read_models(args)

	# Read data
	data, z_init = read_data(args)

	out_dir_copy = args.out_dir

	for i in range(data.shape[0]):

		args.out_dir = out_dir_copy + f"\\{i}\\"
		if not os.path.exists(args.out_dir):
			os.makedirs(args.out_dir)


		# Save original video
		make_gif(data[i].reshape((data.shape[1], args.img_height, args.img_width)), args.out_dir, "original.gif")

		# Re-generate video 
		data_reconstructed = Generate_video(data[i], z_init[i], W, b_h, b_v, Dynamics, Hidden, args, "reconstructed_s_s.gif", True, True)
		data_reconstructed = Generate_video(data[i], z_init[i], W, b_h, b_v, Dynamics, Hidden, args, "reconstructed_s.gif", True, False)
		data_reconstructed = Generate_video(data[i], z_init[i], W, b_h, b_v, Dynamics, Hidden, args, "reconstructed.gif", False, False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_rbm', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\RBM\\weights.pkl", help='Trained RBM model')
    # parser.add_argument('--model_dynamics', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\Dynamics\\FF\\dynamics_model.pt", help='Trained z_i|h_i, z_(i-1)')
    parser.add_argument('--model_dynamics', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\Dynamics\\LSTM\\dynamics_model.pt", help='Trained z_i|h_i, z_(i-1)')
    # parser.add_argument('--model_hidden', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\Generate_Hidden\\FF\\hidden_model.pt", help='Trained h_i|z_i, h_(i-1)')
    parser.add_argument('--model_hidden', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\Generate_Hidden\\LSTM\\hidden_model.pt", help='Trained h_i|z_i, h_(i-1)')
    parser.add_argument('--data', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\Dynamics\\test_frames.pkl", help='Test data frames')
    parser.add_argument('--z', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\Dynamics\\test_z.pkl", help='Test data frames')
    # parser.add_argument('--out_dir', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\Video_Prediction\\FF\\Test\\", help='output directory')
    parser.add_argument('--out_dir', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\Video_Prediction\\LSTM\\Test\\", help='output directory')
    parser.add_argument('--img_height', type=int, default=32, help='Frame height')
    parser.add_argument('--img_width', type=int, default=32, help='Frame width')
    parser.add_argument('--batch_size', type=int, default=16, help='mini batch size')
    parser.add_argument('--epoch', type=int, default=500, help='No. of epochs to train')
    parser.add_argument('--opt', type=str, default="Adam", help='Optimizer : choose from (SGD, ASGD, Adadelta, Adagrad, RMSprop, Adam, AdamW, SparseAdam, Adamax)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 penalty)')
    parser.add_argument('--step', type=int, default=1, help='step size for frame skip')
    parser.add_argument('--re_train', type=bool, default=False, help='Whether to re-train (now using oredicted z as previous z)')

    args = parser.parse_args()

    main(args)   
