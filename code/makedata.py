# %%
# %matplotlib inline

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

torch.manual_seed(1234)

# %% preparing data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_size = 64

transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor()
                             ])

dataroot = "./data"

# this is the attribute for smile
attribute = 31

# %% load testing split and store it to the disk

testset = dset.CelebA(root=dataroot, split='test',
                                        download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)
xtest = torch.Tensor(len(testset), 3, 64, 64)

data = next(iter(testloader))
xtest = data[0]
ytest = data[1][:,attribute]
ytest[ytest==0] = -1

torch.save(xtest, "./data/smile/xtest.pth")
torch.save(ytest, "./data/smile/ytest.pth")

# %% load validation split and store it to the disk

valset = dset.CelebA(root=dataroot, split='valid',
                                        download=True, transform=transform)

valloader = torch.utils.data.DataLoader(valset, batch_size=len(valset), shuffle=False, num_workers=2)
xval = torch.Tensor(len(valset), 3, 64, 64)

data = next(iter(valloader))
xval = data[0]
yval = data[1][:,attribute]
yval[yval==0] = -1

torch.save(xval, "./data/smile/xval.pth")
torch.save(yval, "./data/smile/yval.pth")
# %%
