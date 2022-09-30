import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm

ndim_total = 28 * 28

def one_hot(labels, out=None):
    '''
    Convert LongTensor labels (contains labels 0-9), to a one hot vector.
    Can be done in-place using the out-argument (faster, re-use of GPU memory)
    '''
    if out is None:
        out = torch.zeros(labels.shape[0], 10).to(labels.device)
    else:
        out.zeros_()

    out.scatter_(dim=1, index=labels.view(-1,1), value=1.)
    return out

class MNIST_cINN(nn.Module):
    '''cINN for class-conditional MNISt generation'''
    def __init__(self, lr):
        super().__init__()

        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):

        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, ch_out))

        cond = Ff.ConditionNode(10)
        nodes = [Ff.InputNode(1, 28, 28)]

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, l):
        z, jac = self.cinn(x, c=one_hot(l))
        jac = self.cinn.log_jacobian(run_forward=False)
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn(z, c=one_hot(l), rev=True)
    
#SECTION_START_DATA

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.datasets

batch_size = 256
data_mean = 0.128
data_std = 0.305

    # amplitude for the noise augmentation
augm_sigma = 0.08
data_dir = 'mnist_data'

def unnormalize(x):
    '''go from normaized data x back to the original range'''
    return x * data_std + data_mean


train_data = torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                        transform=T.Compose([T.ToTensor(), lambda x: (x - data_mean) / data_std]))
test_data = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                        transform=T.Compose([T.ToTensor(), lambda x: (x - data_mean) / data_std]))

    # Sample a fixed batch of 1024 validation examples
val_x, val_l = zip(*list(train_data[i] for i in range(1024)))
val_x = torch.stack(val_x, 0).cuda()
val_l = torch.LongTensor(val_l).cuda()

    # Exclude the validation batch from the training data
train_data.data = train_data.data[1024:]
train_data.targets = train_data.targets[1024:]
    # Add the noise-augmentation to the (non-validation) training data:
train_data.transform = T.Compose([train_data.transform, lambda x: x + augm_sigma * torch.randn_like(x)])

train_loader  = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader   = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

#SECTION_END_DATA

#SECTION_START_TRAINING

from time import time

from tqdm import tqdm
import torch
import torch.nn
import torch.optim
import numpy as np

    #import model
    #import data

cinn = MNIST_cINN(5e-4)
cinn.cuda()
scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, milestones=[20, 40], gamma=0.1)

N_epochs = 60
t_start = time()
nll_mean = []

print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
for epoch in range(N_epochs):
    for i, (x, l) in enumerate(train_loader):
        x, l = x.cuda(), l.cuda()
        z, log_j = cinn(x, l)
        print(log_j)

        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / ndim_total
        nll.backward()
        torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 10.)
        nll_mean.append(nll.item())
        cinn.optimizer.step()
        cinn.optimizer.zero_grad()

        if not i % 50:
            with torch.no_grad():
                z, log_j = cinn(val_x, val_l)
                nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / ndim_total

            print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                            i, len(train_loader),
                                                            (time() - t_start)/60.,
                                                            np.mean(nll_mean),
                                                            nll_val.item(),
                                                            cinn.optimizer.param_groups[0]['lr'],
                                                            ), flush=True)
            nll_mean = []
    scheduler.step()

    #torch.save(cinn.state_dict(), 'output/mnist_cinn.pt')
#SECTION_END_TRAINING
