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
    def __init__(self, channels, target_size, n_hidden=None, dropout_rate=0.0):
        super(Net, self).__init__()
        self.target_size = target_size

        # Default n_hidden to number of output values
        if n_hidden is None:
            self.n_hidden = target_size
        else:
            self.n_hidden = n_hidden

        self.conv1 = nn.DataParallel(nn.Conv2d(channels, 96, 3, stride=2))
        self.conv2 = nn.DataParallel(nn.Conv2d(96, 256, 3, stride=2, padding=1))
        self.conv3 = nn.DataParallel(nn.Conv2d(256, 512, 3, stride=2, padding=1))
        self.conv4 = nn.DataParallel(nn.Conv2d(512, 512, 3, stride=2, padding=1))
        self.conv5 = nn.DataParallel(nn.Conv2d(512, 512, 3, stride=2, padding=1))

        self.fc6o = nn.DataParallel(nn.Linear(512 + self.n_hidden, 512))
        self.fc7o = nn.DataParallel(nn.Linear(512, target_size))

        self.fc6h = nn.DataParallel(nn.Linear(512 + self.n_hidden, 512))
        self.fc7h = nn.DataParallel(nn.Linear(512, self.n_hidden))
        
        self.max = nn.MaxPool2d(2)
        self.norm = nn.LocalResponseNorm(channels)

        self.dropout = nn.Dropout(p=dropout_rate)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
    # CNN layer based on ConvNet architecture from Chung et al. (2017): Lip Reading Sentences in the Wild
    def CNN(self, x):
        x = self.norm(self.max(F.relu(self.conv1(x))))
        x = self.norm(self.max(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.dropout(self.conv5(x)))
        return x

    def forward(self, x, hidden):
        x = x.float()
        x = self.CNN(x)                              # Convolutional Layer
        x = x.view(-1, 512)

        combined = torch.cat((x, hidden), 1).to(self.device)

        # fully connected Layers for output
        x = F.relu(self.dropout(self.fc6o(combined)))
        x = F.relu(self.dropout(self.fc7o(x)))

        # Fully connected layers for hidden
        hidden = F.relu(self.fc6h(combined))
        hidden = F.relu(self.fc7h(hidden))

        x = x.view(-1, self.target_size)
        return x, hidden