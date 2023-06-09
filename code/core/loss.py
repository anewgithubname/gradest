import torch
from core.util import comp_dist, comp_median, kernel_comp

def loss(x0, xp, xq, input, sigma = None, kp0 = None, kq0 = None):
    
    if sigma is None:
        sigma = comp_median(xq) 
    
    if kp0 is None:
        kp0 = kernel_comp(xp, x0, sigma) 
    if kq0 is None:
        kq0 = kernel_comp(xq, x0, sigma)
    
    kp01 = kp0 / torch.mean(kp0, 0)
    kp02 = kp0 / torch.mean(kq0, 0)
    kq01 = kq0 / torch.mean(kp0, 0)
    kq02 = kq0 / torch.mean(kq0, 0)
    
    fp0 = torch.matmul(xp, input[:, :-1].T)
    fq0 = torch.matmul(xq, input[:, :-1].T)
        
    l1 = torch.mean(kp01 *fp0, 0) - torch.log(torch.mean(kq01 *torch.exp(fq0), 0))
    l2 = torch.mean(-kq02 *fq0, 0) - torch.log(torch.mean(kp02 *torch.exp(-fp0), 0))
    
    return torch.mean(-l1 - l2)

def logiloss(x0, xp, xq, input, sigma = None, kp0 = None, kq0 = None):
    if sigma is None:
        sigma = comp_median(xq) 
    
    if kp0 is None:
        kp0 = kernel_comp(xp, x0, sigma) 
    if kq0 is None:
        kq0 = kernel_comp(xq, x0, sigma)
    
    fp0 = torch.matmul(xp, input[:, :-1].T)
    fq0 = torch.matmul(xq, input[:, :-1].T)
    
    return torch.mean(torch.mean(kp0 * torch.log(1+torch.exp( - (fp0 + input[:, -1])  )), 0)) \
         + torch.mean(torch.mean(kq0 * torch.log(1+torch.exp(    fq0 + input[:, -1]   )), 0))

def DiscLoss(input, output):
    # # kliep
    # l1 = torch.mean(input[output == 1]) - torch.log(torch.mean(torch.exp(input[output == -1])))
    # l2 = torch.mean(-input[output == -1]) - torch.log(torch.mean(torch.exp(-input[output == 1])))
    # return torch.mean(-l1 - l2)

    # logistic loss
    return torch.mean(torch.log(1+torch.exp(-input[output == 1, :]))) + torch.mean(torch.log(1+torch.exp(input[output == -1, :])))
