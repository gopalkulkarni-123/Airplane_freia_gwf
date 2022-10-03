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
data = torch.tensor(vertices_c)
conditions = torch.tensor(vertices_conditional)
epohs = 50
nll_mean = []

def build_inn():

        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, ch_out))

        cond = Ff.ConditionNode(10)
        nodes = [Ff.InputNode(1, BATCHSIZE, 3)]

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)
    
def forward(x):
    z, jac = cinn(x, c=[conditions])
    jac = cinn.log_jacobian(run_forward=True)
    return z, jac

def reverse_sample(z):
    return cinn(z, c= conditions, rev=True)

    
cinn = build_inn()

trainable_parameters = [p for p in cinn.parameters() if p.requires_grad]
for p in trainable_parameters:
    p.data = 0.01 * torch.randn_like(p)
    
optimizer = torch.optim.Adam(trainable_parameters, lr=5e-4, weight_decay=1e-5)

for i in range(10):
    optimizer.zero_grad()
    # sample data from the moons distribution
    data = torch.tensor(vertices_c)
    conditions = torch.tensor(vertices_conditional)
    #print("data is ",data)
    x = torch.Tensor(data)
    #print("x is", x)
    # pass to INN and get transformed variable z and log Jacobian determinant
    z, log_jac_det = cinn(x, c = [conditions])
    #z, log_jac_det = forward(x)
    # calculate the negative log-likelihood of the model with a standard normal prior
    loss = 0.5*torch.sum(z**2, 1) - log_jac_det
    loss = loss.mean() / N_DIM
    # backpropagate and update the weights
    loss.backward()
    optimizer.step()
