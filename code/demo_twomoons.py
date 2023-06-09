# %%
# %matplotlib inline

import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.utils.data as data_utils
import sbibm
from core.loss import DiscLoss
from core.nn import ResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(999)
torch.cuda.manual_seed(999)
# %%
# setting up the problem 
n = 5000

task_name = "two_moons"
task = sbibm.get_task(task_name)
prior = task.get_prior()
simulator = task.get_simulator()
yobs = task.get_observation(1).to(device)
true_posterior_samples = task.get_reference_posterior_samples(1)
dimtheta = true_posterior_samples.shape[1]
dimy = yobs.shape[1]

# %%

def sample(n, shuffle=False):
    theta = prior(n)
    x = simulator(theta)
    
    if shuffle:
        theta = theta[torch.randperm(n), :]
        x = x[torch.randperm(n), :]
    
    return torch.hstack((theta, x))

# sample data, 5K sample for computing the density ratio, 5k samples for model selection
Xp = sample(n).to(device)
Xpt = sample(n).to(device)

Xq = torch.hstack([Xp[torch.randperm(n), :dimtheta], Xp[torch.randperm(n), dimtheta:]])
Xqt = torch.hstack([Xpt[torch.randperm(n), :dimtheta], Xpt[torch.randperm(n), dimtheta:]])

# get particles from q_0. This is simply particles from p(x), the prior.
Xpos = sample(10000)[:, :dimtheta].to(device)
Xpos = torch.hstack((Xpos, yobs.repeat(Xpos.shape[0],1))).to(device)

# train a network to obtain s(x), the feature map
XTrain = torch.vstack((Xp, Xq))
YTrain = torch.concat((torch.ones((n)), -torch.ones((n))))

DatasetTrain = data_utils.TensorDataset(XTrain, YTrain)
trainLoader = data_utils.DataLoader(DatasetTrain, batch_size=128, shuffle=True)
discNet = ResNet(XTrain.shape[1]).to(device)
discOptimizer = optim.Adam(discNet.parameters(), lr=0.00002, betas=(0.5, 0.999))

print("training the feature map..., may take a few mins")
for epoch in range(200):
    for i, data in enumerate(trainLoader, 0):
        x = data[0].to(device)
        y = data[1].to(device)
        discOptimizer.zero_grad()
        outputs = discNet.forward(x)
        l = DiscLoss(outputs, y)
        l.backward()
        discOptimizer.step()
        
torch.save(discNet.state_dict(), "data/twomoons/discNet_twomoons.pth")
print("feature map trained!")
# %%
# feature map
def s(x):
    return discNet.fea(x)/10
    # return x

# the derivative of feature map
def ds(x, u):
    ret = torch.empty(x.shape, device=device)
    for i in range(u.shape[0]):
        xi = x[i:i+1, ]
        xi.requires_grad = True
        if xi.grad is not None:
            xi.grad.zero_()
        o = sum(discNet.fea(xi)/10, 0)
        o.backward(u[i, ])
        ret[i, ] = xi.grad
    return ret
    
    # return u

# compute feature map for all particles, "f" stands for feature
fXp = s(Xp).detach().cpu().numpy().T
fXpt = s(Xpt).detach().cpu().numpy().T
fXq = s(Xq).detach().cpu().numpy().T
fXqt = s(Xqt).detach().cpu().numpy().T
fPos = s(Xpos).detach().cpu().numpy().T

import numpy as np

wp = np.ones([1, fXp.shape[1]])/fXp.shape[1]
wq = np.ones([1, fXq.shape[1]])/fXq.shape[1]

# %%

# clipping to the unit box
def clip(x):
    return torch.min(torch.max(x, -torch.ones(x.shape).to(device)), torch.ones(x.shape).to(device))
    # return x

import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('matlab/'))
eng.parpool("threads")
print("matlab engine started!")

from scipy.io.matlab import savemat
savemat("data/twomoons/true.mat", {"true": true_posterior_samples.detach().cpu().numpy()})

for t in range(150):
    print("iteration: ", t)
    
    # the scatter plot for transported particles
    fig = plt.figure()
    plt.scatter(Xpos[:, 0].detach().cpu().numpy(), Xpos[:, 1].detach().cpu().numpy(), c='r', marker='x', label='transported particles')
    plt.scatter(true_posterior_samples[:n, 0].detach().cpu().numpy(), true_posterior_samples[:n, 1].detach().cpu().numpy(), c='g', marker='x', label='true posterior sample')
    plt.legend()
    fig.savefig(f"figs/sbi/twomoons/{t}.png")
    plt.show()
    plt.close(fig)
    
    # the histogram for transported particles and features
    fig = plt.figure()
    plt.hist(fXp[0, :], bins=20, alpha=0.5, label='p_tilde(s(x)_1)')
    plt.hist(fXq[0, :], bins=20, alpha=0.5, label='q_tilde(s(x)_1)')
    plt.legend()
    fig.savefig(f"figs/sbi/twomoons/hist_{t}.png")
    plt.show()
    plt.close(fig)


    # compute the median 
    sigma = eng.comp_med(fXq)
    sigma_list = sigma * np.array([2, 4, 8])

    # calculate nabla_t s(t) 
    u, t_list = eng.vg_update(np.hstack((fXq,fXqt,fPos)), fXp, fXq, wp, wq, fXpt, fXqt, sigma_list, nargout=2)
    u = torch.tensor(u).float().T.to(device)
    print("t_list: ", t_list)
    
    # computing the update
    print("computing the update...")
    update = ds(torch.vstack((Xq, Xqt, Xpos)), u)
    update = update / torch.max(torch.sqrt(torch.sum(update ** 2, dim = 1)))

    # clip out of bounds particles
    Xq[:, :dimtheta] = clip(Xq[:, :dimtheta] + .1 * update[:Xq.shape[0], :dimtheta])
    Xqt[:, :dimtheta] = clip(Xqt[:, :dimtheta] + .1 * update[Xq.shape[0]:-Xpos.shape[0], :dimtheta])
    Xpos[:, :dimtheta] = clip(Xpos[:, :dimtheta] + .1 * update[-Xpos.shape[0]:, :dimtheta])

    # copy back to cpu
    fXq = s(Xq).detach().cpu().numpy().T
    fXqt = s(Xqt).detach().cpu().numpy().T
    fPos = s(Xpos).detach().cpu().numpy().T
    
    Xq = Xq.detach()
    Xqt = Xqt.detach()
    Xpos = Xpos.detach()
    
    # save the current particles to disk for matlab to plot
    savemat(f"data/twomoons/xp_{t}.mat", {"xp": Xp.detach().cpu().numpy()})
    savemat(f"data/twomoons/xq_{t}.mat", {"xq": Xq.detach().cpu().numpy()})
    savemat(f"data/twomoons/xpos_{t}.mat", {"xpos": Xpos.detach().cpu().numpy()})
    

# %%
