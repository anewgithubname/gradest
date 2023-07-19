# %%

import torch
device = torch.device("cuda:0" if torch.device else "cpu")
# device = torch.device("cpu")
torch.set_default_device(device)

from IPython import display

from core.torchGradFlow import infer_cv
from core.nn import NPNet
from core.util import comp_median
from torch import ones, zeros, eye

# %%
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import numpy as np

err = []
d = 2
mu = ones(d, device=device)*1
covar = eye(d, device=device)*.5

# %%
error_over_n = []
for n in [500, 1000, 2500, 5000]:
    err = []
    for seed in range(11): 
        print(n, seed)
        torch.manual_seed(seed)
        # generate multivariate normal distribution with mean mu and covariance sigma
        Xp = MVN(mu, covar).sample((n,)).to(device)
        Xq = MVN(zeros(d), eye(d)).sample((n,)).to(device)

        sigma = comp_median(Xq)
        sigmalist = [sigma * .75, sigma * 1, sigma * 2]

        grad = infer_cv(Xp, Xp, Xq, sigmalist)
        
        Xp_ = Xp.clone().to(device)
        # Xp_.requires_grad = True
        # tt = MVN(mu, covar).log_prob(Xp_) - MVN(zeros(d, device=device), eye(d, device=device)).log_prob(Xp_)
        # grad_Xp = torch.autograd.grad(tt.sum(), Xp_)[0]
        grad_Xp = -(Xp_ - mu).matmul(torch.linalg.inv(covar)) + Xp_.matmul(torch.linalg.inv(eye(d, device=device)))
        
        e = torch.mean(torch.sum((grad - grad_Xp)**2, 1))
        print("error:", e.item())
        err.append(e.item())
        
        print("")

    error_over_n.append(err)

#convert the list of lists to a numpy array
error_over_n = np.array(error_over_n)
np.save(f'error_over_n_{mu[0]}.npy', error_over_n)

# %%
# load the data

import matplotlib.pyplot as plt

error_over_n = np.load(f'error_over_n_{mu[0]}.npy')
# plot errorbar for each n
plt.errorbar([500, 1000, 2500, 5000], np.mean(error_over_n, 1), yerr=np.std(error_over_n, 1), fmt='-o')

plt.xlabel('n')
plt.ylabel('MSE')

print(np.mean(error_over_n, 1))
print(np.std(error_over_n, 1))
# %%
