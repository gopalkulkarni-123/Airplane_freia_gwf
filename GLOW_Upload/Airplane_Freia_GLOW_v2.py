import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from Airplane_data_v1 import vertices_c,vertices_conditional
from time import time
import numpy as np

condition_ = torch.tensor(vertices_conditional)
data = torch.tensor(vertices_c)

class cINN(nn.Module):
    
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

        cond = Ff.ConditionNode(8)
        nodes = [Ff.InputNode(110667, 3)]

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x):
        z, jac = self.cinn(x, c=condition_)
        jac = self.cinn.log_jacobian(run_forward=False)
        return z, jac

    def reverse_sample(self, z, l):
        #condition_ = torch.tensor(vertices_conditional)
        return self.cinn(z, c=condition_, rev=True)
    
cinn = cINN(5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, milestones=[20, 40], gamma=0.1)

N_epochs = 60
t_start = time()
nll_mean = []

print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
for epoch in range(N_epochs):
    for i, (x) in enumerate(data):
        z, log_j = cinn(x, c = condition_)
        print(log_j)

        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / 3
        nll.backward()
        torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 10.)
        nll_mean.append(nll.item())
        cinn.optimizer.step()
        cinn.optimizer.zero_grad()

        if not i % 50:
            with torch.no_grad():
                z, log_j = cinn(data.val_x, data.val_l)
                nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / 3

            print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                            i, len(data.train_loader),
                                                            (time() - t_start)/60.,
                                                            np.mean(nll_mean),
                                                            nll_val.item(),
                                                            cinn.optimizer.param_groups[0]['lr'],
                                                            ), flush=True)
            nll_mean = []
    scheduler.step()