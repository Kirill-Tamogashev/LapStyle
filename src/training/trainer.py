import torch
from torch import Tensor

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.training.losses import LapStyleLoss
from src.model.descriminator import Descriminator
from src.model.drafting_network import DraftingNetwork
from src.model.revision_network import RevisionNetwork


class Trainer:
    def __init__(self, learning_rate, ):
        self.revision_network = RevisionNetwork()
        self.drafting_network = DraftingNetwork()
        self.descriminator = Descriminator()
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss = LapStyleLoss()
        
        #### code for feature extraction ####
        self.vgg_layers = models.vgg19(pretrained=True).features
        for param in self.self.vgg_layers.parameters():
            param.requires_grad = False
        self.capture_layers = [0,5,10, 19, 28]
        
    def train(self):
        for _ in range(n_epochs):
            for batch in self.dataloader:
                content, style = batch
                content, style = content.to(device), style.to(device)
                
                
    

        
        
        