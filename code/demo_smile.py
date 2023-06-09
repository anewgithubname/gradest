# %%
# %matplotlib inline

# loading packages
import torch
import torchvision.utils as vutils

from core.loss import loss as klloss
from core.nn import NPNet, CNNDiscNet
from torch import optim

from core.util import comp_median, kernel_comp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

# %%

# estimate gradient of log r(x) w.r.t. x
def gradest(x0, xp, xq, sigma):
    m = x0.shape[1]
    net = NPNet(x0.shape[0], m).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    kp0 = kernel_comp(xp, x0, sigma) 
    kq0 = kernel_comp(xq, x0, sigma)
    
    for i in range(5000):
        optimizer.zero_grad()
        loss = klloss(x0, xp, xq, net.forward(None), sigma=sigma, kp0=kp0, kq0=kq0)
        loss.backward()
        optimizer.step()
            
    return net.forward(None)

# perform model selection for sigma and use the selected sigma to estimate the gradient
def infer(x0, xp, xq, xpt, xqt, sigma_list):
    m = x0.shape[1]
    x0_modelsel = torch.vstack((xpt, xqt))
    s = []
    for sigma in sigma_list:
        para = gradest(x0_modelsel, xp, xq, sigma)
        logr0 = torch.sum(para[:, :m] * x0_modelsel  + para[:, m:m+1], 1)
        logr0_pq = logr0[:xpt.shape[0]] - torch.log(torch.mean(torch.exp(logr0[xpt.shape[0]:]), 0))
        logr0_qp = -logr0[xpt.shape[0]:] - torch.log(torch.mean(torch.exp(-logr0[:xpt.shape[0]]), 0))
        testloss = -torch.mean(logr0_pq) - torch.mean(logr0_qp)

        print("tloss:", testloss.item())
        s.append(testloss.item())
    
    print(s)
    return gradest(x0, xp, xq, sigma_list[np.argmin(s)])[:, :m]

torch.manual_seed(1234)
# %%

import matplotlib.pyplot as plt
import numpy as np
# %%

# loading the pretrained feature map
PATH = './data/smile/smile_net.pth'
discNet = CNNDiscNet()
discNet.load_state_dict(torch.load(PATH))
discNet.eval()
discNet.to(device)

# feature map
def s(x):
    return discNet.fea(x)/10

# gradient of feature map
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

# gradient of the prediction function
def df0(x):
    ret = torch.empty(x.shape)
    for i in range(x.shape[0]):
        xi = x[i:i+1, ]
        xi.requires_grad = True
        if xi.grad is not None:
            xi.grad.zero_()
        # forward, not fea. 
        o = sum(discNet.forward(xi), 0)
        o.backward()
        ret[i, ] = xi.grad
    return ret

# %% 
# now we use the validation set to calibrate the gradient 
# and use the test set to evaluate the performance

from scipy.io.matlab import savemat
showFigure = True

# we are only using the first 2000 images from the validation set
# and the first 500 images from the test set to save time
nval = 2000
ntest = 500

# you would need to run "makedata.py" first to generate these data
xval = torch.load("data/smile/xval.pth")[:nval].to(device)
yval = torch.load("data/smile/yval.pth")[:nval].to(device)
xtest = torch.load("data/smile/xtest.pth")[:ntest].to(device)
ytest = torch.load("data/smile/ytest.pth")[:ntest].to(device)

# x0 is our testing image that are not smiling
x0 = xtest[ytest==-1, ]
x1 = xtest[ytest==1, ]

# change this singlestep to true if you want to run the single step update experiment
singlestep = False
if singlestep:
    lr = .56
    # single step
    MAX_ITER = 1 
else:
    lr = .01
    MAX_ITER = 500
    
for t in range(MAX_ITER):
    
    print("iteration {}".format(t))

    if showFigure:
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Sample Images")
        plt.imshow(np.transpose(vutils.make_grid(x0.to(device)[:64], padding=2).cpu(),(1,2,0)))
        plt.savefig("figs/smile/img{}.png".format(t))
        plt.show()
    
    fxval = s(xval)
    savemat("data/smile/fx{}.mat".format(t), {"fx1":fxval[yval==1, ].detach().cpu().numpy(), "fx0":fxval[yval==-1, ].detach().cpu().numpy()})

    if singlestep:
        update = df0(torch.vstack((x0, xval[yval==-1, ]))).to(device)
    else:
        # we use the first 500 smiling and non-smiling images to train the gradient model 
        fxp = fxval[yval==1, ][:500,].detach()
        fxq = fxval[yval==-1, ][:500,].detach()
        # we use the rest of the images to do model selection
        fxp_t = fxval[yval==1, ][500:,].detach()
        fxq_t = fxval[yval==-1, ][500:,].detach()
        
        fx0 = s(torch.vstack((x0, xval[yval==-1, ]))).detach()
        
        med = comp_median(fxq)
        sigma_list = [med*.5, med*1, med*2]
        u = infer(fx0, fxp, fxq, fxp_t, fxq_t, sigma_list)
        
        # computing the nabla_s log r(s) nabla_x s(x)
        update = ds(torch.vstack((x0, xval[yval==-1, ])), u)
        
    # update = update / torch.max(torch.sqrt(torch.sum(update ** 2, dim = (1,2,3))))
    # update testing images that are not smiling
    x0 = x0 + lr*update[:x0.shape[0], ]
    # update the validation images that are not smiling
    xval[yval==-1, ] = xval[yval==-1, ] + lr*update[x0.shape[0]:, ]
        
    if showFigure:
        plt.figure(figsize=(8,8))
        plt.axis("off")
        uplot = plt.imshow(np.transpose(vutils.make_grid(torch.abs(update).to(device)[:64], padding=2, normalize=False).cpu(),(1,2,0)))
        plt.savefig("figs/smile/grad{}.png".format(t))
        plt.show()
    
    torch.save(x0, "data/smile/x0")
    

# %%
print("final outcome:")
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Sample Images")
plt.imshow(np.transpose(vutils.make_grid(x0.to(device)[:64], padding=2).cpu(),(1,2,0)))
plt.show()
# %%
