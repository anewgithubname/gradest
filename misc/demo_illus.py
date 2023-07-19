# %%

import torch

from core.torchGradFlow import infer_cv, plot_norm_contour
from core.util import comp_median
from torch import ones, zeros, eye
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import numpy as np

err = []
d = 2
mu = zeros(d, device=device)+1
covar = eye(d, device=device)*.5
n = 1000

# %%

seed = 1
torch.manual_seed(seed)

# generate multivariate normal distribution with mean mu and covariance sigma
Xp = MVN(mu, covar).sample((n,)).to(device)
Xq = MVN(zeros(d), eye(d)).sample((n,)).to(device)

Xqt = MVN(zeros(d), eye(d)).sample((n,)).to(device)
Xqt.requires_grad = True
logrq = MVN(mu, covar).log_prob(Xqt) - MVN(zeros(d, device=device), eye(d, device=device)).log_prob(Xqt)
grad_logrq = torch.autograd.grad(logrq.sum(), Xqt)[0]

med = comp_median(Xq)
sigma_list = [med * .75, med * 1, med * 2]
grad = infer_cv(Xqt.detach(), Xp, Xq, sigma_list=sigma_list)

print("estimation error:", torch.mean(torch.sum((grad - grad_logrq)**2,1)).item())

# %%
import matplotlib.pyplot as plt

# generate a grid [-5, 5]
x = torch.linspace(-2, 2, 20)
y = torch.linspace(-2, 2, 20)
x0 = torch.stack(torch.meshgrid(x, y), dim=-1).reshape(-1, 2).to(device)

grad = infer_cv(x0, Xp, Xq, sigma_list=sigma_list)

# %%
# generate a grid [-5, 5]
plt.figure(figsize=(5, 5))
plot_norm_contour(mu.cpu(), covar.cpu())
plot_norm_contour(zeros(d).cpu(), eye(d).cpu(), 'b')
plt.title("red: p, blue q, green: estimated grad, red: true gradient")

x0.requires_grad = True
logr0 = MVN(mu, covar).log_prob(x0) - MVN(zeros(d, device=device), eye(d, device=device)).log_prob(x0)
grad_logr0 = torch.autograd.grad(logr0.sum(), x0)[0]

#plot the gradient estimate
plt.quiver(x0[:, 0].detach().cpu(), x0[:, 1].detach().cpu(), 
           grad[:, 0].detach().cpu(), 
           grad[:, 1].detach().cpu(), scale=100, color='g')

plt.quiver(x0[:, 0].detach().cpu(), x0[:, 1].detach().cpu(), 
           grad_logr0[:, 0].detach().cpu(),
           grad_logr0[:, 1].detach().cpu(), scale=100, color='r')

# %%
