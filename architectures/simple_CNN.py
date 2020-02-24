import torch
import torch.nn as nn
import torch.nn.functional as F

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class Net(nn.Module):
    def __init__(self, channels, target_size):
        super(Net, self).__init__()
        self.target_size = target_size
        self.conv1 = nn.DataParallel(nn.Conv2d(channels, 16, 3))
        self.pool = nn.DataParallel(nn.MaxPool2d(5, 5))
        self.conv2 = nn.DataParallel(nn.Conv2d(16, 48, 3))
        self.fc1 = nn.Linear(432, 256)
        self.fc2 = nn.Linear(256, target_size)
        
    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 432)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x.view(-1, self.target_size), dim=1)
        return x