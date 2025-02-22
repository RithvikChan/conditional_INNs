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


print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
for epoch in range(N_epochs):
    print(len(data.train_loader))
    for i, (x, l) in enumerate(data.train_loader):
        x, l = x.cuda(), l.cuda()
        z, log_j, outs = cinn(x, l)
        
        

        # Train with reverse
        # randval = 1.0 * torch.randn(256, model.ndim_total).cuda()
        # synth_x = cinn.reverse_sample(out, l)[0]
        # z, log_j = cinn(synth_x, l)
        # print(log_j.size())
        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total
        nll.backward()

        out, forward_log_j, outs_rev_removed = invert_cinn(x, l)
        synth_x, log_j_rev, outs_rev = invert_cinn.reverse_sample(out, l)
        z_rev, log_j_rev_forw, _ = invert_cinn(synth_x, l)
        
        # print("REV::") 
        # print(list(outs_rev.keys())[::-1])
        # print("REM::")
        # print(list(outs_rev_removed.keys()))
        
        if i==229:
            mse_list = []
            interm_keys = list(outs_rev_removed.keys())
            print(interm_keys[:15])
            for interm_val in range(0,15):
                first_interms = outs_rev[interm_keys[interm_val]].cpu().detach().numpy().flatten()
                second_interms = outs_rev_removed[interm_keys[interm_val]].cpu().detach().numpy().flatten()
                mse = np.sum(np.square(np.subtract(first_interms, second_interms)))/len(first_interms)
                mse_list.append(mse)
            print(mse_list)
            plt.plot([str(i) for i in list(range(0,15))], mse_list, label =str(epoch))
            plt.legend()
            plt.savefig("analysis.png")

        inver_nll = torch.mean(z_rev**2) / 2 - torch.mean(log_j_rev_forw) / model.ndim_total
        inver_nll.backward()


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
