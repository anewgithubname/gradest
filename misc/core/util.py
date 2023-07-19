
import torch

def comp_dist(x,y):
    t1 = torch.tile(torch.sum(x**2, dim=1, keepdim=True), (1, y.shape[0]))
    t2 = -2*torch.matmul(x, y.T)
    t3 = torch.tile(torch.sum(y**2, dim=1, keepdim=True).T, (x.shape[0], 1))
    return t1 + t2 + t3

def comp_median(x):
    return torch.sqrt(.5*comp_dist(x, x).flatten().median())

def kernel_comp(x, y, sigma):
    # compute a kernel matrix
    t1 = torch.tile(torch.sum(x**2, dim=1, keepdim=True), (1, y.shape[0]))
    t2 = -2*torch.matmul(x, y.T)
    t3 = torch.tile(torch.sum(y**2, dim=1, keepdim=True).T, (x.shape[0], 1))
    return torch.exp(- (t1 + t2 + t3)/2/(sigma**2))