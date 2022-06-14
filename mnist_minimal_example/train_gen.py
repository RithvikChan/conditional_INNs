from audioop import reverse
from time import time
import sys 
from tqdm import tqdm
import torch
import torch.nn
import torch.optim
import numpy as np

import model
import data
import matplotlib.pyplot as plt

invert_cinn = model.MNIST_cINN(5e-4)
cinn = model.MNIST_cINN(5e-4)
cinn.cuda()
invert_cinn.cuda()
scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, milestones=[20, 40], gamma=0.1)
invert_scheduler = torch.optim.lr_scheduler.MultiStepLR(invert_cinn.optimizer, milestones=[20, 40], gamma=0.1)
N_epochs = 100
t_start = time()
nll_mean = []
nll_mean_rev = []
datas = []
labels = []

print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
# for epoch in range(N_epochs):
#     print(len(data.train_loader))
for i, (x, l) in enumerate(data.train_loader):
    
    # print(i)
    x, l = x.cuda(), l.cuda()
    out, forward_log_j, outs_rev_removed = invert_cinn(x, l)
    synth_x, log_j_rev, outs_rev = invert_cinn.reverse_sample(out, l)
    z_rev, log_j_rev_forw, _ = invert_cinn(synth_x, l)
    inver_nll = torch.mean(z_rev**2) / 2 - torch.mean(log_j_rev_forw) / model.ndim_total
    inver_nll.backward()
    
    gen_x = synth_x
    datas.append(gen_x.detach().cpu().numpy())
    # print(gen_x.size())
    
    # print(l[0].item())
    labels.append(l.detach().cpu().numpy())
    #Reverse
    torch.nn.utils.clip_grad_norm_(invert_cinn.trainable_parameters, 10.)


    invert_cinn.optimizer.step()
    invert_cinn.optimizer.zero_grad()

print(len(datas))
print(len(labels))
np.save('mnist_data', datas)
np.save('labels', labels)

    # with torch.no_grad():
    #     z, log_j, interms = cinn(data.val_x, data.val_l)
    #     nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total

    # with torch.no_grad():
    #     z_rev, log_j_rev, interms_rev = invert_cinn(data.val_x, data.val_l)
    #     nll_val_rev = torch.mean(z_rev**2) / 2 - torch.mean(log_j_rev) / model.ndim_total

    # print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
    #                                                 i, len(data.train_loader),
    #                                                 (time() - t_start)/60.,
    #                                                 np.mean(nll_mean),
    #                                                 nll_val.item(),
    #                                                 cinn.optimizer.param_groups[0]['lr'],
    #                                                 ), flush=True)
    # print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
    #                                                 i, len(data.train_loader),
    #                                                 (time() - t_start)/60.,
    #                                                 np.mean(nll_mean_rev),
    #                                                 nll_val_rev.item(),
    #                                                 invert_cinn.optimizer.param_groups[0]['lr'],
    #                                                 ), flush=True)
    # nll_mean = []
    # scheduler.step()
    # nll_mean_rev = []
    # invert_scheduler.step()

# torch.save(cinn.state_dict(), 'output/mnist_cinn.pt')
