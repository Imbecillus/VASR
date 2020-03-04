import torch
import torch.nn as nn
import torch.nn.functional as F

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

# Based on ConvNet architecture from Chung et al. (2017): Lip Reading Sentences in the Wild

class Net(nn.Module):
    def __init__(self, channels, target_size, dropout_rate = 0.0):
        super(Net, self).__init__()
        self.target_size = target_size
        self.conv1 = nn.Conv2d(channels, 96, 3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, target_size)
        self.dropout = nn.Dropout(p=dropout_rate)

        # For parallel processing
        self.layer1 = nn.DataParallel(self.conv1)
        self.layer2 = nn.DataParallel(self.conv2)
        self.layer3 = nn.DataParallel(self.conv3)
        self.layer4 = nn.DataParallel(self.conv4)
        self.layer5 = nn.DataParallel(self.conv5)
        self.layer6 = nn.DataParallel(self.fc6)
        self.layer7 = nn.DataParallel(self.fc7)
        
        self.max = nn.MaxPool2d(2)
        self.norm = nn.LocalResponseNorm(channels)
        
    def forward(self, x):
        x = x.float()
        x = self.norm(self.max(F.relu(self.layer1(x))))
        x = self.norm(self.max(F.relu(self.layer2(x))))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.dropout(self.layer5(x)))
        x = x.view(-1, 512)
        x = F.relu(self.dropout(self.layer6(x)))
        x = F.relu(self.layer7(x))
        x = F.softmax(x.view(-1, self.target_size), dim=1)
        return x