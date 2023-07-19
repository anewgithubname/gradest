import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.fc1 = nn.Linear(m, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, m+1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    # def fea(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    
class Net2(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.fc1 = nn.Linear(d, 200)
        self.fc2 = nn.Linear(200, m)
        self.fc3 = nn.Linear(m, 200)
        self.fc4 = nn.Linear(200, m+1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
    def fea(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Net3(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.fc1 = nn.Linear(d, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, d+1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
    def fea(self, x):
        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # return x
        
        return torch.flatten(x, 1)
    
class CNNDiscNet2(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(512, m)
        self.fc2 = nn.Linear(m, 64)
        self.fc3 = nn.Linear(64, m+1)

    def forward(self, x):
        x = F.leaky_relu(self.maxpool1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool3(self.conv3(x)), 0.2, inplace=True)
        
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        x = self.fc3(x)
        return x

    def fea(self, x):
        x = F.leaky_relu(self.maxpool1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool3(self.conv3(x)), 0.2, inplace=True)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    

class LogiNet(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.fc1 = nn.Linear(m, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, m+1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
class NPNet(nn.Module):
    def __init__(self, theta0):
        super().__init__()
        self.para = nn.Parameter(theta0)

    def forward(self, x):
        return self.para
    
class DiscNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 5)
        self.fc4 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return self.fc4(x)

    def fea(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)
    
class ResNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, d)
        self.fc4 = nn.Linear(d, 1)

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x += identity
        return self.fc4(x)

    def fea(self, x):
        identity = x
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x += identity
        return x
    
class CNNDiscNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        # self.norm2 = nn.BatchNorm2d(64)
        # self.norm3 = nn.BatchNorm2d(128)
        # self.norm4 = nn.BatchNorm2d(256)
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.leaky_relu(self.maxpool1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool4(self.conv4(x)), 0.2, inplace=True)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        x = self.fc3(x)
        return x

    def fea(self, x):
        x = F.leaky_relu(self.maxpool1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool4(self.conv4(x)), 0.2, inplace=True)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x