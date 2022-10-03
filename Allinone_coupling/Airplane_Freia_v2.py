# standard imports
import torch
import torch.nn as nn
from airplane_data import vertices_c
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

BATCHSIZE = 100
N_DIM = 3

# we define a subnet for use inside an affine coupling block
# for more detailed information see the full tutorial
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(),
                         nn.Linear(512,  dims_out))

# a simple chain of operations is collected by ReversibleSequential
inn = Ff.SequenceINN(N_DIM)
for k in range(8):
    inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)

# a very basic training loop
for i in range(1):
    optimizer.zero_grad()
    # sample data from the moons distribution
    data = vertices_c
    #print("data is ",data)
    x = torch.Tensor(data)
    #print("x is", x)
    # pass to INN and get transformed variable z and log Jacobian determinant
    z, log_jac_det = inn(x)
    # calculate the negative log-likelihood of the model with a standard normal prior
    loss = 0.5*torch.sum(z**2, 1) - log_jac_det
    loss = loss.mean() / N_DIM
    # backpropagate and update the weights
    loss.backward()
    optimizer.step()

# sample from the INN by sampling from a standard normal and transforming
# it in the reverse direction
z = torch.randn(BATCHSIZE, N_DIM)
samples, _ = inn(z, rev=True)
sampled_data = samples.detach().numpy()


#Plotting
x = sampled_data[:,0]
x_1 = vertices_c[:,0]

y = sampled_data[:,1]
y_1 = vertices_c[:,1]

z = sampled_data[:,2]
z_1 = vertices_c[:,2]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

#ax.scatter(x_1,y_1,z_1)
ax.scatter(x,y,z)

plt.show()
