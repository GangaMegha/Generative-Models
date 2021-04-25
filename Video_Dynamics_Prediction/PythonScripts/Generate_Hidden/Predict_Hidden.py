import numpy as np
import string
import argparse
import csv
import pandas as pd

import pickle
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from Models import LSTM, FF

import matplotlib.pyplot as plt


def read_data(dir_path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    train_p_h_v = pickle.load(open(f"{dir_path}\\train_p_h_v.pkl" , 'rb' ) )
    test_p_h_v = pickle.load(open(f"{dir_path}\\test_p_h_v.pkl" , 'rb' ) )

    train_z = pickle.load(open(f"{dir_path}\\train_z.pkl" , 'rb' ) )
    test_z = pickle.load(open(f"{dir_path}\\test_z.pkl" , 'rb' ) )

    return train_p_h_v, train_z, test_p_h_v, test_z


def plot_loss(train_loss, val_loss, out_dir, title):
    fig = plt.figure()
    plt.plot(train_loss, c='r', label="Train")
    plt.plot(val_loss, c='g', label="Val")
    plt.legend()
    plt.title("MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.savefig(out_dir + f"\\{title}")



def retrain_Inference(model, loss_func, test_p_h_v, test_z, batch_size):

    max_len_test = test_p_h_v.shape[0]
    frames = test_p_h_v.shape[1]
    img_size = test_p_h_v.shape[2]
    z_size = test_z.shape[2]

    with torch.no_grad():

        test_loss = 0

        i = 0
        while(i<max_len_test):

            flag = True

            if(i+batch_size)>max_len_test:
                batch_size = max_len_test-i
                prev_h = torch.tensor(test_p_h_v[i:,0].reshape((batch_size, 1, img_size)), dtype=torch.float32)
            else:
                prev_h = torch.tensor(test_p_h_v[i:i+batch_size,0].reshape((batch_size, 1, img_size)), dtype=torch.float32)

            for j in range(1, frames):

                X = test_z[i:i+batch_size, j].reshape((batch_size, 1, z_size))
                y = test_p_h_v[i:i+batch_size, j].reshape((batch_size, 1, img_size))

                # Step 2. Prepare inputs
                input_video = torch.tensor(X, dtype=torch.float32)
                targets = torch.tensor(y, dtype=torch.float32)

                # Step 3. Run our forward pass.
                out = model(input_video, prev_h, flag)

                # Step 4. Compute loss
                loss = loss_func(out, targets)
                test_loss +=loss.item()

                prev_h = out.clone()

                flag=False

            i += batch_size

    return test_loss/max_len_test


def Inference(model, loss_func, test_p_h_v, test_z, batch_size):

    max_len_test = test_p_h_v.shape[0]
    frames = test_p_h_v.shape[1]

    with torch.no_grad():

        test_loss = 0

        i = 0
        while(i<max_len_test):

            if(i+batch_size)>max_len_test:
                X = test_z[i:]
                y = test_p_h_v[i:]
            else:
                X = test_z[i:i+batch_size]
                y = test_p_h_v[i:i+batch_size]

            # Step 2. Prepare inputs
            input_video = torch.tensor(X[:, 1:], dtype=torch.float32)
            prev_h = torch.tensor(y[:, :-1], dtype=torch.float32)

            targets = torch.tensor(y[:, 1:], dtype=torch.float32)
            torch.autograd.set_detect_anomaly(True)

            # print(input_video.size())
            # print(prev_h.size())
            # print(targets.size())
            # Step 3. Run our forward pass.
            out = model(input_video, prev_h, True)

            # Step 4. Compute loss
            loss = loss_func(out, targets)
            test_loss +=loss.item()

            i += batch_size

    return test_loss/max_len_test
               
def re_Train(model, optimizer, loss_func, train_p_h_v, train_z, val_p_h_v, val_z, out_dir, batch_size, epochs):
    train_loss_list = []
    val_loss_list = []

    min_loss = 1e8
    Flag = False

    max_len_train = len(train_z)
    max_len_val = len(val_z)

    frames = train_p_h_v.shape[1]
    img_size = train_p_h_v.shape[2]
    z_size = train_z.shape[2]

    best_epoch = 0

    print("\n\nStaring the re-training.....")

    # Train Model
    for epoch in range(epochs):  
        train_loss = 0
        val_loss = 0

        i = 0
        while(i<max_len_train):

            flag = True

            if(i+batch_size)>max_len_train:
                new_batch_size = max_len_train-i
                prev_h = torch.tensor(train_p_h_v[i:,0].reshape((new_batch_size, 1, img_size)), dtype=torch.float32)
            else:
                new_batch_size = batch_size
                prev_h = torch.tensor(train_p_h_v[i:i+new_batch_size,0].reshape((new_batch_size, 1, img_size)), dtype=torch.float32)

            for j in range(1, frames):

                # Step 1. Pytorch accumulates gradients. Clear them out before each instance
                model.zero_grad()


                # Step 2. Prepare inputs
                X = train_p_h_v[i:i+new_batch_size, j].reshape((new_batch_size, 1, z_size))
                y = train_z[i:i+new_batch_size, j].reshape((new_batch_size, 1, img_size))

                input_video = torch.tensor(X, dtype=torch.float32)
                targets = torch.tensor(y, dtype=torch.float32)

                torch.autograd.set_detect_anomaly(True)

                # Step 3. Run our forward pass.
                out = model(input_video, prev_h, flag)

                # Step 4. Compute loss
                loss = loss_func(out, targets)
                train_loss +=loss.item()

                prev_h = out.clone()

                # Step 5. Compute the gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward(retain_graph=True)
                optimizer.step()

                flag=False

            i += batch_size

        train_loss = train_loss/max_len_train
            
        train_loss_list.append(train_loss)

        val_loss = retrain_Inference(model, loss_func, val_p_h_v, val_z, batch_size)
        val_loss_list.append(val_loss)


        print(f"\n\nEpoch {epoch+1} ------> loss - Train : {train_loss}, Val : {val_loss}")


        if val_loss < min_loss : 
            torch.save(model, f"{out_dir}\\hidden_model_retrained.pt")
            Flag = True
            min_loss = val_loss
            best_epoch = epoch
            print("\n\nNew min_loss found : ", min_loss)

    plot_loss(train_loss_list, val_loss_list, out_dir, "Retrain_Loss_plot.png")

    # Load best model
    if Flag :
        print("\n\n\n Minimum val loss found at epoch : ", best_epoch+1)
        print(f"Loading the best model from {out_dir}\\hidden_model_retrained.pt")
        model = torch.load(f"{out_dir}\\hidden_model_retrained.pt")
        print(model)

    return model

def Train(model, optimizer, loss_func, train_p_h_v, train_z, val_p_h_v, val_z, out_dir, batch_size, epochs):
    train_loss_list = []
    val_loss_list = []

    min_loss = 1e8
    Flag = False
    max_len_train = len(train_z)
    max_len_val = len(val_z)
    best_epoch = 0

    # Train Model
    for epoch in range(epochs):  
        train_loss = 0
        val_loss = 0

        i = 0
        while(i<max_len_train):
            # Step 1. Pytorch accumulates gradients. Clear them out before each instance
            model.zero_grad()

            if(i+batch_size)>max_len_train:
                X = train_z[i:]
                y = train_p_h_v[i:]
            else:
                X = train_z[i:i+batch_size]
                y = train_p_h_v[i:i+batch_size]

            # Step 2. Prepare inputs
            input_video = torch.tensor(X[:, 1:], dtype=torch.float32)
            prev_h = torch.tensor(y[:, :-1], dtype=torch.float32)

            targets = torch.tensor(y[:, 1:], dtype=torch.float32)
            torch.autograd.set_detect_anomaly(True)

            # print(input_video.size())
            # print(prev_h.size())
            # print(targets.size())
            # Step 3. Run our forward pass.
            out = model(input_video, prev_h, True)

            # Step 4. Compute loss
            loss = loss_func(out, targets)
            train_loss +=loss.item()

            # Step 5. Compute the gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

            i += batch_size

        train_loss = train_loss/max_len_train
            
        train_loss_list.append(train_loss)

        val_loss = Inference(model, loss_func, val_p_h_v, val_z, batch_size)
        val_loss_list.append(val_loss)


        print(f"\n\nEpoch {epoch+1} ------> loss - Train : {train_loss}, Val : {val_loss}")


        if val_loss < min_loss : 
            torch.save(model, f"{out_dir}\\hidden_model.pt")
            Flag = True
            min_loss = val_loss
            best_epoch = epoch
            print("\n\nNew min_loss found : ", min_loss)

    plot_loss(train_loss_list, val_loss_list, out_dir, "Loss_plot.png")

    # Load best model
    if Flag :
        print("\n\n\n Minimum val loss found at epoch : ", best_epoch+1)
        print(f"Loading the best model from {out_dir}\\hidden_model.pt")
        model = torch.load(f"{out_dir}\\hidden_model.pt")
        print(model)

    return model


def main(args):

    # read the dataset
    train_p_h_v, train_z, test_p_h_v, test_z = read_data(args.in_dir)

    # Scaling probabilities to reflect more loss
    train_p_h_v = train_p_h_v
    test_p_h_v = test_p_h_v

    # Create output directory
    args.out_dir = args.out_dir + "\\" + str(args.option) + "\\"
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # train_p_h_v, train_z, test_p_h_v, test_z = train_p_h_v[:20], train_z[:20], test_p_h_v[:20], test_z[:20]

    # Splitting train data into train and val
    frac = int(0.15*len(train_z))
    val_p_h_v = train_p_h_v[:frac]
    train_p_h_v = train_p_h_v[frac:]

    val_z = train_z[:frac]
    train_z = train_z[frac:]

    print("\n\n\nTrain set size : ", train_p_h_v.shape)
    print("Validation set size : ", val_p_h_v.shape)
    print("Test set size : ", test_p_h_v.shape)


    # Create model
    if args.re_train:
        print("\n\nLoading pre-trained model...")
        model = torch.load(f"{args.out_dir}\\dynamics_model.pt")
    else :
        if args.option=="LSTM":
            model = LSTM(embed_dim=train_z.shape[2], output_dim=train_p_h_v.shape[2], hidden_dim=32, n_layers=1, drop_prob=0.5)
        elif args.option=="FF":
            model = FF(embed_dim=train_z.shape[2], output_dim=train_p_h_v.shape[2], drop_prob=0.5)
        else:
            print("\n\n\t\tInvalid option. Please choose from (LSTM, FF)")
            return

    # Define Opimizer
    params = model.parameters()
    if args.opt.lower() == "sgd" : optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    elif args.opt.lower() == "asgd" : optimizer = optim.ASGD(params, lr=args.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=args.weight_decay)
    elif args.opt.lower() == "adadelta" : optimizer = optim.Adadelta(params, lr=args.lr, rho=0.9, eps=1e-06, weight_decay=args.weight_decay)
    elif args.opt.lower() == "adagrad" : optimizer = optim.Adagrad(params, lr=args.lr, lr_decay=0, weight_decay=args.weight_decay, initial_accumulator_value=0, eps=1e-10)
    elif args.opt.lower() == "rmsprop" : optimizer = optim.RMSprop(params, lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=args.momentum, centered=False)
    elif args.opt.lower() == "adam" : optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    elif args.opt.lower() == "adamw" : optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    elif args.opt.lower() == "sparseadam" : optimizer = optim.SparseAdam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    elif args.opt.lower() == "adamax" : optimizer = optim.Adamax(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else :
        print("\t\tOptimizer", args.opt, "does not correspond to any known setup\n\n\n")
        print("\t\tPlease choose from { SGD, ASGD, Adadelta, Adagrad, RMSprop, Adam, AdamW, SparseAdam, Adamax }")
        return

    # Loss function for regression
    loss_func = nn.MSELoss()

    if args.re_train:
        model = re_Train(model, optimizer, loss_func, train_p_h_v, train_z, val_p_h_v, val_z, args.out_dir, args.batch_size, args.epoch)
        test_loss = retrain_Inference(model, loss_func, test_p_h_v, test_z, args.batch_size)
        print(f"Retrain Inference on Test data gives loss = {test_loss}")
    else:
        model = Train(model, optimizer, loss_func, train_p_h_v, train_z, val_p_h_v, val_z, args.out_dir, args.batch_size, args.epoch)

        test_loss = Inference(model, loss_func, test_p_h_v, test_z, args.batch_size)
        print(f"Inference on Test data gives loss = {test_loss}")

        test_loss = retrain_Inference(model, loss_func, test_p_h_v, test_z, args.batch_size)
        print(f"Retrain Inference on Test data gives loss = {test_loss}")
    

   


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\Generate_Hidden\\", help='folder containing train_p_h_v.pkl and test_p_h_v.pkl')
    parser.add_argument('--out_dir', type=str, default="C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\Generate_Hidden\\", help='output directory')
    parser.add_argument('--option', type=str, default="LSTM", help='Option to run (LSTM, FF)')
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


