from typing import List

import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class LapStyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_1 = 16
        self.lambda_2 = 7
        self.alpha = 5
        
        self.style_loss = style_loss
        self.content_loss = content_loss
        self.preseptual_loss = preseptual_loss
        self.mean_variance_loss = mean_variance_loss
        
        #### code for feature extraction ####
        self.vgg_layers = models.vgg19(pretrained=True).features
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.capture_layers = [0, 5, 10, 19, 28]
        
    def forward(
        self, 
        generated: List[Tensor], 
        true_style: List[Tensor], 
        true_content: List[Tensor],
    ) -> float:
     
        l_cnt, l_stl, l_m_v, l_per = 0, 0, 0, 0
        for i, layer in enumerate(self.vgg_layers):
            generated = layer(generated)
            true_style = layer(true_style)
            true_content = layer(true_content)
            if i in self.capture_layers:
                l_per += self.preseptual_loss(generated, true_content) / 5
                l_m_v += self.mean_variance_loss(generated, true_style) / 5       
            if i in {10, 19}:
                l_cnt += self.content_loss(generated, true_content) / 2
                l_stl +=  self.style_loss(generated, true_style) / 2
                
        total_content_loss = l_cnt + self.lambda_1 * l_per
        total_style_loss = l_stl + self.lambda_2 * l_m_v
                        
        return total_content_loss + self.alpha * total_style_loss
        

def content_loss(generated: Tensor, true_content: Tensor) -> Tensor:
    return F.mse_loss(generated, true_content)


def style_loss(generated: Tensor, true_style: Tensor) -> Tensor:
    batch_size, n_channels, height, width = generated.size()
    assert true_style.size(0) == batch_size
    
    generated = generated.view(batch_size, n_channels, height * width)
    true_style = true_style.view(batch_size, n_channels, height * width)
    
    G = torch.einsum("kij,kjm -> kim", generated, generated.permute(0, 2, 1))
    A = torch.einsum("kij,kjm -> kim", true_style, true_style.permute(0, 2, 1))
    return torch.mean((G-A)**2)


def preseptual_loss(generated: Tensor, true_content: Tensor):
    generated = F.normalize(generated)
    true_content = F.normalize(true_content)
    return F.mse_loss(generated, true_content)
    

def mean_variance_loss(generated: Tensor, true_style: Tensor) -> Tensor:
    
    batch, n_channels, *_ = generated.size()
    
    mean_style = torch.mean(true_style, dim=(1, 2))
    std_style = torch.std(true_style.reshape(n_channels, -1), dim=0)
    
    mean_generated = torch.mean(generated, dim=(1, 2))
    std_generated  = torch.std(generated.reshape(n_channels, -1), dim=0)

    loss = torch.abs(mean_generated - mean_style).mean()
    loss += torch.abs(std_generated - std_style).mean()
    return loss
