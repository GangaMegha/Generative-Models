import pickle
import numpy as np 
import matplotlib.pyplot as plt

in_dir = "C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\Dynamics\\"
out_dir = "C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Outputs\\Dynamics\\"



train_z = pickle.load(open(f"{in_dir}\\train_z.pkl" , 'rb' ) )

# z for each ball
z_b_1 = train_z[0, :, :4]
z_b_2 = train_z[0, :, 4:8]
z_b_3 = train_z[0, :, 8:]


# Plot z_1
fig1 = plt.figure()
plt.plot(z_b_1[:, 0], c='r', label="ball 1")
plt.plot(z_b_2[:, 0], c='b', label="ball 2")
plt.plot(z_b_3[:, 0], c='g', label="ball 3")
plt.legend()
plt.title("Dynamics : z_1")
plt.xlabel("frames")
plt.ylabel("z_1")
plt.savefig(out_dir + "\\z_1.png")


# Plot z_2
fig1 = plt.figure()
plt.plot(z_b_1[:, 1], c='r', label="ball 1")
plt.plot(z_b_2[:, 1], c='b', label="ball 2")
plt.plot(z_b_3[:, 1], c='g', label="ball 3")
plt.legend()
plt.title("Dynamics : z_2")
plt.xlabel("frames")
plt.ylabel("z_2")
plt.savefig(out_dir + "\\z_2.png")


# Plot z_3
fig1 = plt.figure()
plt.plot(z_b_1[:, 2], c='r', label="ball 1")
plt.plot(z_b_2[:, 2], c='b', label="ball 2")
plt.plot(z_b_3[:, 2], c='g', label="ball 3")
plt.legend()
plt.title("Dynamics : z_3")
plt.xlabel("frames")
plt.ylabel("z_3")
plt.savefig(out_dir + "\\z_3.png")


# Plot z_4
fig1 = plt.figure()
plt.plot(z_b_1[:, 3], c='r', label="ball 1")
plt.plot(z_b_2[:, 3], c='b', label="ball 2")
plt.plot(z_b_3[:, 3], c='g', label="ball 3")
plt.legend()
plt.title("Dynamics : z_4")
plt.xlabel("frames")
plt.ylabel("z_4")
plt.savefig(out_dir + "\\z_4.png")


# Plot z_1 v/s z2
fig1 = plt.figure()
plt.plot(z_b_1[:, 1], z_b_1[:, 0], c='r', label="ball 1")
plt.plot(z_b_2[:, 1], z_b_2[:, 0], c='b', label="ball 2")
plt.plot(z_b_3[:, 1], z_b_3[:, 0], c='g', label="ball 3")
plt.legend()
plt.title("Dynamics : z_pos")
plt.xlabel("z_x")
plt.ylabel("z_y")
plt.savefig(out_dir + "\\z_pos.png")


in_dir = f"C:\\Users\\ganga\\Github\\Generative-Models\\Project\\Data\\"
with open(f"{in_dir}\\billiards_train.pkl", 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data["coord_lim"])

# for i in range(data["coord_lim"].shape[0]):

# 	print(data["coord_lim"])