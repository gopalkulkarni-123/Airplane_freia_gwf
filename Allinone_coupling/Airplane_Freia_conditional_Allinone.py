import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from Airplane_data_v1 import vertices_c,vertices_conditional

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

BATCHSIZE = int(vertices_c.size/3)
N_DIM = (3,)
cond_dims =(11,)

#Hyperparameters
hidden_layers = 512
coup_blks = 8
epochs = 10

# we define a subnet for use inside an affine coupling block
# for more detailed information see the full tutorial
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, hidden_layers), nn.ReLU(),
                         nn.Linear(hidden_layers,  dims_out))
    
# a simple chain of operations is collected by ReversibleSequential
inn = Ff.SequenceINN(*N_DIM)
for k in range(coup_blks):
    inn.append(Fm.AllInOneBlock,cond = 0, cond_shape = cond_dims, subnet_constructor=subnet_fc)

optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)

# a very basic training loop
for i in range(epochs):
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
       
#Remove the following section to run the code
''' This program is an attempt to generate a conditional input that follows the example provided in the documentation of FREIA.
It uses Fm.AllInOneBlock which is found to be working in the previous case. The only change from the previous version is a conditional input
The error encountered is as follows
Traceback (most recent call last):
  File "e:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\Airplane\Airplane_Freia_v8_conditional.py", line 38, in <module>
    z, log_jac_det = inn(x, c = [conditions])
  File "C:\Users\Gopal Kulkarni\anaconda3\lib\site-packages\torch\nn\modules\module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "e:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\Airplane\FrEIA\framework\sequence_inn.py", line 106, in forward
    x_or_z, j = self.module_list[i](x_or_z, c=[c[self.conditions[i]]],
  File "C:\Users\Gopal Kulkarni\anaconda3\lib\site-packages\torch\nn\modules\module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "e:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\Airplane\FrEIA\modules\all_in_one_block.py", line 248, in forward
    a1 = self.subnet(x1c)
  File "C:\Users\Gopal Kulkarni\anaconda3\lib\site-packages\torch\nn\modules\module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "C:\Users\Gopal Kulkarni\anaconda3\lib\site-packages\torch\nn\modules\container.py", line 139, in forward
    input = module(input)
  File "C:\Users\Gopal Kulkarni\anaconda3\lib\site-packages\torch\nn\modules\module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "C:\Users\Gopal Kulkarni\anaconda3\lib\site-packages\torch\nn\modules\linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: expected scalar type Float but found Double
'''