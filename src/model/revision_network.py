import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(size, size, kernel_size=3, padding='same'), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(size, size, kernel_size=1)
        )
    
    def forward(self, x):
        return x + self.block(x)


class RevisionNetwork(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Upsample(size=self.img_size),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1)
        )
    
    def forward(self, draft, contur):
        input_ = torch.cat([draft, contur], dim=1)
        return F.interpolate(self.net(input_), size=self.img_size)
        
    