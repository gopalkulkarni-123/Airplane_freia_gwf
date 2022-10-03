import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from Airplane_data_v1 import vertices_c,vertices_conditional

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

BATCHSIZE = int(vertices_c.size/3)
N_DIM = 3
cond_dims =(11,)

# we define a subnet for use inside an affine coupling block
# for more detailed information see the full tutorial
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(),
                         nn.Linear(512,  dims_out))
    
# a simple chain of operations is collected by ReversibleSequential
#inn = Ff.SequenceINN(N_DIM)
cond = Ff.ConditionNode(8)
nodes = [Ff.InputNode(1, BATCHSIZE,3)]
for k in range(8):
    nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))
    nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,{'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))

optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)

# a very basic training loop
for i in range(10):
    optimizer.zero_grad()
    # sample data from the moons distribution
    data = torch.tensor(vertices_c)
    conditions = torch.tensor(vertices_conditional)
    #print("data is ",data)
    x = torch.Tensor(data)
    #print("x is", x)
    # pass to INN and get transformed variable z and log Jacobian determinant
    z, log_jac_det = inn(x, c = [conditions])
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
with open(r'E:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\ShapeNet\airplane_cond.txt', 'w') as k:
    for i in range (1000):
       k.write("["+str(sampled_data[i][0])+","+str(sampled_data[i][1])+","+str(sampled_data[i][2])+"],"+"\n")