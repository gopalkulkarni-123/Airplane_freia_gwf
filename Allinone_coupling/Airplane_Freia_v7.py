import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from Airplane_data_v1 import vertices_c,vertices_conditional

#input_dims = 3
BATCHSIZE = int(vertices_c.size/3)
cond_dims = 10
input_dims = (BATCHSIZE,)


def subnet_fc(dims_in, dims_out):
    '''Return a feed-forward subnetwork, to be used in the coupling blocks below'''
    return nn.Sequential(nn.Linear(dims_in, 128), nn.ReLU(),
                         nn.Linear(128,  128), nn.ReLU(),
                         nn.Linear(128,  dims_out))

# a tuple of the input data dimension. 784 is the dimension of flattened MNIST images.
# (input_dims would be (3, 32, 32) for CIFAR for instance)


# use the input_dims (784,) from above
cinn = Ff.SequenceINN(*input_dims)

for k in range(8):
    # The cond=0 argument tells the operation which of the conditions it should
    # use, that are supplied with the call. So cond=0 means 'use the first condition'
    # (there is only one condition in this case).
    cinn.append(Fm.AllInOneBlock, cond=0, cond_shape=cond_dims, subnet_constructor=subnet_fc)

optimizer = torch.optim.Adam(cinn.parameters(), lr=0.001)

# the conditions have to be given as a list (in this example, a list with
# one entry, 'one_hot_labels').  In general, multiple conditions can be
# given. The cond argument of the append() method above specifies which
# condition is used for each operation.
#z, jac = cinn(x, c=[one_hot_labels])
for i in range(100):
    optimizer.zero_grad()
    # sample data from the moons distribution
    data = vertices_c
    #print("data is ",data)
    x = torch.Tensor(data)
    #print("x is", x)
    # pass to INN and get transformed variable z and log Jacobian determinant
    z, log_jac_det = cinn(x)
    # calculate the negative log-likelihood of the model with a standard normal prior
    loss = 0.5*torch.sum(z**2, 1) - log_jac_det
    loss = loss.mean() / input_dims
    # backpropagate and update the weights
    loss.backward()
    optimizer.step()

# sample from the INN by sampling from a standard normal and transforming
# it in the reverse direction
z = torch.randn(BATCHSIZE, input_dims)
samples, _ = cinn(z,c = [vertices_conditional], rev=True)
sampled_data = samples.detach().numpy()
with open(r'E:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\ShapeNet\airplane_data_sampled_v5.txt', 'w') as k:
    for i in range (1000):
       k.write("["+str(sampled_data[i][0])+","+str(sampled_data[i][1])+","+str(sampled_data[i][2])+"],"+"\n")