import torch
import torch.nn as nn
import torch.nn.functional as F

class residual_block(torch.nn.Module):
    def __init__(self, channels, filters, kernel_size, stride, device, project_shortcut=False, skip_bn=False):
        super(residual_block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.project_shortcut = project_shortcut
        self.skip_bn = skip_bn

        self.conv2d_1 = torch.nn.Conv2d(channels, filters, kernel_size, stride, padding=1).to(device)
        self.batch_norm_1 = torch.nn.BatchNorm2d(channels).to(device)
        self.batch_norm_2 = torch.nn.BatchNorm2d(filters).to(device)
        self.ReLU = torch.nn.ReLU().to(device)
        self.conv2d_2 = torch.nn.Conv2d(filters, filters, kernel_size, stride=1, padding=1).to(device)

        self.projection_shortcut = torch.nn.Conv2d(channels, filters, (1, 1), stride, bias=False).to(device)

    def forward(self, inputs):
        shortcut = inputs

        if self.skip_bn is False:
            inputs = self.batch_norm_1(inputs)
            inputs = self.ReLU(inputs)

        if self.project_shortcut:
            shortcut = self.projection_shortcut(shortcut)
        
        inputs = self.conv2d_1(inputs)
        inputs = self.batch_norm_2(inputs)
        inputs = self.ReLU(inputs)
        inputs = self.conv2d_2(inputs)

        return inputs + shortcut

class ResNet(torch.nn.Module):
    def __init__(self, channels, cnn_dense_units, cnn_filters, dropout_rate, device):
        super(ResNet, self).__init__()
        self.cnn_dense_units = cnn_dense_units
        self.ReLU = torch.nn.ReLU().to(device)
        
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(channels, cnn_filters[0], (3, 3), padding=1), torch.nn.BatchNorm2d(cnn_filters[0])).to(device)
        
        layers = []
        layers.append(residual_block(cnn_filters[0], cnn_filters[0], (3, 3), 1, device, project_shortcut=False, skip_bn=True))
        for ix, num_filters in enumerate(cnn_filters[1:]):
            layers.append(residual_block(cnn_filters[ix], num_filters, (3, 3), 2, device, project_shortcut=True).to(device))
        self.layers = torch.nn.Sequential(*layers)

        self.final = torch.nn.Conv2d(cnn_filters[-1], cnn_dense_units, (5, 5), 1).to(device)
        
    def forward(self, inputs):
        flow = inputs.float()
        # Layer 1
        flow = self.layer1(flow)
        flow = self.ReLU(flow)

        # Layers 2-...
        flow = self.layers(flow)
        
        # Final ResNet-Layer
        flow = self.final(flow)
        flow = self.ReLU(flow)

        return flow

class Net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_dim, num_layers, embedding_layer, bidirectional = False):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding_layer = embedding_layer

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)

        directions = 1 if not bidirectional else 2
        self.hidden2tag = nn.Linear(hidden_dim * directions, target_dim)

    def forward(self, flow):
        try:
            flow = self.embedding_layer(flow)
            flow = torch.squeeze(flow, dim=3).transpose(1, 2)
            flow, _ = self.lstm(flow)
            flow = self.hidden2tag(flow.squeeze())
            #tag_scores = F.log_softmax(tag_space)
            return flow
        except BaseException as e:
            print(flow.shape)
            print(flow)
            print(e)
            