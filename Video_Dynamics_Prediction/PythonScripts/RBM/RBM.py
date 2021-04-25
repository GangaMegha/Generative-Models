import numpy as np


class RBM():
    def __init__(self, num_hidden, num_visible, lr, n, batch_size, epochs):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.lr = lr
        self.n = n
        self.batch_size = batch_size
        self.epochs = epochs

        self.W = np.random.randn(num_hidden, num_visible)/np.sqrt(0.5*(num_visible + num_hidden)) # weights

        self.b_h = np.zeros((num_hidden, 1)) # bias latent
        self.b_v = np.zeros((num_visible, 1)) # bias visible

        self.dW = []
        self.db_h = []
        self.db_v = []

    def sigmoid(self, x):  
        #Sigmoid activation 
        #Implemented interms  of tanh for increased stability
        return .5 * (1 + np.tanh(.5 * x))

    
    def bernoulli_array(self, prob_array, dim):
        # Simulating Bernoulli from uniform
        sample = np.zeros(dim)

        # Draw x~Uni[0,1]
        uni_sample = np.random.uniform(0, 1, dim)

        # return 1 if x < p else return 0
        diff = uni_sample - prob_array
        coords = np.argwhere(diff<0)
        sample[[*coords.T]] = 1  

        return sample

    def gibbs_sampling(self, h_0):

        h = h_0.copy()

        for i in range(self.n):

            # (v x h) @ (h x b) + (v x 1) = (v x b)
            p_v_h = self.sigmoid(self.W.T @ h + self.b_v)
            v = self.bernoulli_array(p_v_h, (p_v_h.shape[0], p_v_h.shape[1]))

            # (h x v) @ (v x b) + (h x 1) = (h x b)
            p_h_v = self.sigmoid(self.W @ v + self.b_h)
            h = self.bernoulli_array(p_h_v, (p_h_v.shape[0], p_h_v.shape[1]))

        return v, h, p_h_v

    def gradient_descent(self, v_0, p_h_v_0, v_n, p_h_v_n):

        # Compute the gradients
        # (h x b) @ (b x v) - (h x b) @ (b x v) = (h x v)
        self.dW = (p_h_v_0 @ v_0 - p_h_v_n @ v_n)/self.batch_size
        self.db_h = np.mean(p_h_v_0 - p_h_v_n, axis=1)[:, np.newaxis]
        self.db_v = np.mean(v_0 - v_n, axis=0)[:, np.newaxis]
        
        # Weight update
        self.W   = self.W   + self.lr * self.dW
        self.b_h = self.b_h + self.lr * self.db_h
        self.b_v = self.b_v + self.lr * self.db_v


    def reconstruction_error(self, v):
        # Sample hidden state
        p_h_v = self.sigmoid(self.W @ v + self.b_h)
        h = self.bernoulli_array(p_h_v, (p_h_v.shape[0], p_h_v.shape[1]))

        # Sample viible state
        p_v_h = self.sigmoid(self.W.T @ h + self.b_v)
        v_sampled = self.bernoulli_array(p_v_h, (p_v_h.shape[0], p_v_h.shape[1]))    

        return np.sum(np.mean((v-v_sampled)**2, axis=1), axis=0)


    def reconstruct_image(self, v):
        # Sample hidden state
        p_h_v = self.sigmoid(self.W @ v + self.b_h)
        h = self.bernoulli_array(p_h_v, (p_h_v.shape[0], p_h_v.shape[1]))

        # Sample viible state
        p_v_h = self.sigmoid(self.W.T @ h + self.b_v)
        v_sampled = self.bernoulli_array(p_v_h, (p_v_h.shape[0], p_v_h.shape[1]))    

        return v_sampled


    def Train(self, train, val):

        num_batches = int(train.shape[0]/self.batch_size)
        train_loss = []
        val_loss = []

        for epoch in range(self.epochs):

            # Shuffling the data
            train = np.random.permutation(train)

            # Splitting data into batches
            batches = np.array_split(train, num_batches)

            for i in range(num_batches):

                # visible units from data
                v_0 = batches[i].T

                # (h x v) @ (v x b) + (h x 1) = (h x b)
                p_h_v_0 = self.sigmoid(self.W @ v_0 + self.b_h)
                h_0 = self.bernoulli_array(p_h_v_0, (p_h_v_0.shape[0], p_h_v_0.shape[1]))

                # Run the markov chain
                v_n, h_n, p_h_v_n = self.gibbs_sampling(h_0)

                # Compute gradients
                self.gradient_descent(v_0.T, p_h_v_0, v_n.T, p_h_v_n)

            # Compute reconstruction errror
            error_train = self.reconstruction_error(train.T)
            error_val = self.reconstruction_error(val.T)

            print(f"Epoch {epoch+1} ------> Error => Train : {error_train}, Val : {error_val}")
 
            train_loss.append(error_train)
            val_loss.append(error_val)

        return train_loss, val_loss