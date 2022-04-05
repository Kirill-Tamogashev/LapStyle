import torch
import torch.nn as nn


class DescriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class Descriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            DescriminatorBlock(3, 32, 3, 2),
            DescriminatorBlock(32, 32, 3, 2),
            DescriminatorBlock(32, 32, 3, 2),
            DescriminatorBlock(32, 32, 3, 2),
            DescriminatorBlock(32, 32, 3, 2),
        )
        
        self.linear = nn.Linear(15 * 15 * 32, 1)
    
    def forward(self, inputs):
        hidden = self.net(inputs)
        hidden = hidden.reshape(hidden.size(0), -1)
        return self.linear(hidden)
        