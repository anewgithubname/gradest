# %%

from core.gradest import infer

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
def infer_cpp(Xp, Xq, X):
    return torch.from_numpy(infer(Xp.detach().cpu().numpy(), Xq.detach().cpu().numpy(), X.detach().cpu().numpy())).to(device)

# %%
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import numpy as np

err = []
d = 2
mup = zeros(d, device=device)
covarp = eye(d, device=device)

muq = zeros(d, device=device)
covarq = eye(d, device=device)*2

# %%
ntrial = 23
error_over_n = []
nsamples = [500, 1000, 2500, 5000]

# %%
for n in nsamples:
    err = []
    for seed in range(ntrial): 
        print(n, seed)
        torch.manual_seed(seed)
        # generate multivariate normal distribution with mean mu and covariance sigma
        Xp = MVN(mup, covarp).sample((n,)).to(device)
        Xq = MVN(muq, covarq).sample((n,)).to(device)

        # sigma = comp_median(Xq)
        # sigmalist = [sigma * .75, sigma * 1, sigma * 2]

        Xpt = MVN(mup, covarp).sample((10000,)).to(device)
        # grad = infer_cv(Xp_, Xp, Xq, sigmalist)
        grad = infer_cpp(Xp, Xq, Xpt)
        
        Xpt.requires_grad = True
        tt = MVN(mup, covarp).log_prob(Xpt) - MVN(muq, covarq).log_prob(Xpt)
        grad_Xpt = torch.autograd.grad(tt.sum(), Xpt)[0]
        
        e = torch.mean(torch.sum((grad - grad_Xpt)**2, 1))
        print("error:", e.item())
        err.append(e.item())
        
        print("")

    error_over_n.append(err)

#convert the list of lists to a numpy array
error_over_n = np.array(error_over_n)
np.save(f'error_over_n_{mup[0]}.npy', error_over_n)

# %%
# load the data

import matplotlib.pyplot as plt

error_over_n = np.load(f'error_over_n_{mup[0]}.npy')
# plot errorbar for each n
plt.errorbar(nsamples, np.mean(error_over_n, 1), yerr=np.std(error_over_n, 1)/np.sqrt(ntrial), fmt='-o')

plt.xlabel('n')
plt.ylabel('MSE')

print(np.mean(error_over_n, 1))
print(np.std(error_over_n, 1))
 # %%
