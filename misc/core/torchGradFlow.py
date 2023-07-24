# %%
import torch
from core.loss import loss as klloss
from core.nn import NPNet
from torch import optim

from core.util import comp_median, kernel_comp

import numpy as np

# %%

def plot_norm_contour(mu, sigma, c = 'r'):
    from matplotlib import pyplot as plt
    from numpy import linspace, meshgrid, empty
    from scipy.stats import multivariate_normal

    # Create a grid of points on which to evaluate the distribution
    x = linspace(mu[0] - 10*sigma[0,0], mu[0] + 10*sigma[0,0], 100)
    y = linspace(mu[1] - 10*sigma[1,1], mu[1] + 10*sigma[1,1], 100)
    X, Y = meshgrid(x, y)

    # Create a 2D standard normal distribution with mean mu and standard deviation sigma
    pos = empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(mu, sigma)

    # Evaluate the distribution on the grid of points
    Z = rv.pdf(pos)

    # Create the contour plot
    plt.contour(X, Y, Z, levels=5, colors = c, alpha = 1)

    plt.xlabel('X')
    plt.ylabel('Y')


# %%

# estimate gradient of log r(x) w.r.t. x using sigma
def gradest(x0, xp, xq, sigma):
    net = NPNet(torch.zeros(x0.shape[0], x0.shape[1]+1))
    optimizer = optim.Adagrad(net.parameters(), lr=1e-1)
    
    kp0 = kernel_comp(xp, x0, sigma) 
    kq0 = kernel_comp(xq, x0, sigma)
    
    old_para = net.forward(x0).clone().detach()
    for i in range(2000):
                
        optimizer.zero_grad()
        loss = klloss(x0, xp, xq, net.forward(x0), sigma=sigma, kp0=kp0, kq0=kq0)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            newpara = net.forward(x0).clone().detach()        
            # compare with old para
            if (newpara- old_para).norm() < 1e-2:
                print("break at", i)
                break
        
            old_para = newpara
        
    return net.forward(x0)

# x0: on which to estimate the gradient
# xp: numerator samples
# xq: denominator samples
# sigma_list: a list of sigma to select from
def infer_cv(x0, xp, xq, sigma_list):
    m = x0.shape[1]
    #shuffle xp and xq 
    xp = xp[torch.randperm(xp.shape[0]), ]
    xq = xq[torch.randperm(xq.shape[0]), ]
    s = []
    for sigma in sigma_list:
        testloss = 0
        for k in range(5):
            #select the k-th fold of xp as the test set
            xptk = xp[k*int(xp.shape[0]/5):(k+1)*int(xp.shape[0]/5), ]
            xqtk = xq[k*int(xq.shape[0]/5):(k+1)*int(xq.shape[0]/5), ]
            #select the rest of xp as the training set
            xpk = torch.vstack((xp[:k*int(xp.shape[0]/5), ], xp[(k+1)*int(xp.shape[0]/5):, ]))
            xqk = torch.vstack((xq[:k*int(xq.shape[0]/5), ], xq[(k+1)*int(xq.shape[0]/5):, ]))

            x0_modelsel = torch.vstack((xptk, xqtk))
            para = gradest(x0_modelsel, xpk, xqk, sigma)
            
            logr0 = torch.sum(para[:, :m] * x0_modelsel  + para[:, m:m+1], 1)
            logr0_pq = logr0[:xptk.shape[0]] - torch.log(torch.mean(torch.exp(logr0[xptk.shape[0]:]), 0))
            logr0_qp = -logr0[xptk.shape[0]:] - torch.log(torch.mean(torch.exp(-logr0[:xptk.shape[0]]), 0))
            t = -torch.mean(logr0_pq) - torch.mean(logr0_qp)
            testloss = testloss + t/5
            
            print("k:", k, f'sigma: {sigma.item():.2f}', "testloss:", t.item())
        
        print("tloss:", testloss.item())
        s.append(testloss.item())
        
    print(s); s = torch.tensor(s)
    return gradest(x0, xp, xq, sigma_list[torch.argmin(s)])[:, :m]

