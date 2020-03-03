import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

def Net(channels, classes, dropout_rate):
    # Defining DropoutBlock class, which refers to BasicBlock for all calculations, but adds a dropout layer on forward pass
    class DropoutBlock(models.resnet.BasicBlock):
        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
            super().__init__(inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None)
            self.dropout = nn.Dropout2d(dropout_rate)
            
        def forward(self, x):
            return self.dropout(super().forward(x)) 

    model = models.ResNet(DropoutBlock, [2, 2, 2, 2], num_classes=classes)

    model.fc = nn.Linear(512, classes)

    return model
