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

loss_list_orig = []
loss_list_synth = []


print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
for epoch in range(N_epochs):
    print(len(data.train_loader))
    for i, (x, l) in enumerate(data.train_loader):

        x, l = x.cuda(), l.cuda()
        out, forward_log_j, _ = invert_cinn(x, l)
        synth_x, log_j_rev, outs_rev = invert_cinn.reverse_sample(out, l)

        print("SIZE:", synth_x.size())

        new_synth_x = torch.detach(synth_x)
        

        z, log_j, outs = cinn(new_synth_x, l)


        inver_nll = torch.mean(out**2) / 2 - torch.mean(forward_log_j) / model.ndim_total
        inver_nll.backward()

        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total
        nll.backward()

        
        #Normal
        torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 10.)

        #Reverse
        torch.nn.utils.clip_grad_norm_(invert_cinn.trainable_parameters, 10.)

        nll_mean.append(nll.item())
        nll_mean_rev.append(inver_nll.item())

        cinn.optimizer.step()
        cinn.optimizer.zero_grad()

        invert_cinn.optimizer.step()
        invert_cinn.optimizer.zero_grad()


    loss_list_orig.append(np.mean(nll_mean_rev))
    loss_list_synth.append(np.mean(nll_mean))

    if epoch == 10:
        plt.plot([str(i) for i in list(range(0,epoch+1))], loss_list_orig, label ='Original')
        plt.plot([str(i) for i in list(range(0,epoch+1))], loss_list_synth, label ='Synth')
        plt.legend()
        plt.savefig("analysis_loss.png")
    with torch.no_grad():
        z, log_j, interms = cinn(data.val_x, data.val_l)
        nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total

    with torch.no_grad():
        z_rev, log_j_rev, interms_rev = invert_cinn(data.val_x, data.val_l)
        nll_val_rev = torch.mean(z_rev**2) / 2 - torch.mean(log_j_rev) / model.ndim_total

    print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                    i, len(data.train_loader),
                                                    (time() - t_start)/60.,
                                                    np.mean(nll_mean),
                                                    nll_val.item(),
                                                    cinn.optimizer.param_groups[0]['lr'],
                                                    ), flush=True)
    print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                    i, len(data.train_loader),
                                                    (time() - t_start)/60.,
                                                    np.mean(nll_mean_rev),
                                                    nll_val_rev.item(),
                                                    invert_cinn.optimizer.param_groups[0]['lr'],
                                                    ), flush=True)
    nll_mean = []
    scheduler.step()
    nll_mean_rev = []
    invert_scheduler.step()

torch.save(cinn.state_dict(), 'output/mnist_cinn.pt')
