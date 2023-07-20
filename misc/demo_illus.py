# %%
from core.gradest import infer

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.set_default_device(device)

from core.torchGradFlow import infer_cv, plot_norm_contour
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

n = 10000

# %%

seed = 1
torch.manual_seed(seed)

# generate multivariate normal distribution with mean mu and covariance sigma
Xp = MVN(mup, covarp).sample((n,)).to(device)
Xq = MVN(muq, covarq).sample((n,)).to(device)

Xpt = MVN(mup, covarp).sample((10000,)).to(device)
Xpt.requires_grad = True
logrpt = MVN(mup, covarp).log_prob(Xpt) - MVN(muq, covarq).log_prob(Xpt)
grad_logrq = torch.autograd.grad(logrpt.sum(), Xpt)[0]

# med = comp_median(Xq)
# sigma_list = [med * .75, med * 1, med * 2]
# grad = infer_cv(Xqt.detach(), Xp, Xq, sigma_list=sigma_list)
grad = infer_cpp(Xp, Xq, Xpt)

print("estimation error:", torch.mean(torch.sum((grad - grad_logrq)**2,1)).item())

# %%
import matplotlib.pyplot as plt

# generate a grid [-5, 5]
x = torch.linspace(-2, 2, 20)
y = torch.linspace(-2, 2, 20)
x0 = torch.stack(torch.meshgrid(x, y), dim=-1).reshape(-1, 2).to(device)

# grad = infer_cv(x0, Xp, Xq, sigma_list=sigma_list)
grad = infer_cpp(Xp, Xq, x0)

# %%
# generate a grid [-5, 5]
plt.figure(figsize=(5, 5))
plot_norm_contour(mup.cpu(), covarp.cpu())
plot_norm_contour(muq.cpu(), covarq.cpu(), 'b')
plt.title("red: p, blue q, green: estimated grad, red: true gradient")

x0.requires_grad = True
logr0 = MVN(mup, covarp).log_prob(x0) - MVN(muq, covarq).log_prob(x0)
grad_logr0 = torch.autograd.grad(logr0.sum(), x0)[0]

#plot the gradient estimate
plt.quiver(x0[:, 0].detach().cpu(), x0[:, 1].detach().cpu(), 
           grad[:, 0].detach().cpu(), 
           grad[:, 1].detach().cpu(), scale=40, color='g')

plt.quiver(x0[:, 0].detach().cpu(), x0[:, 1].detach().cpu(), 
           grad_logr0[:, 0].detach().cpu(),
           grad_logr0[:, 1].detach().cpu(), scale=40, color='r')

plt.xlim(-2, 2)
plt.ylim(-2, 2)

# %%
