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
    def __init__(self):
        #### models ####
        self.drafting_network = DraftingNetwork()
        self.revision_network = RevisionNetwork()
        self.descriminator = Descriminator()
        
        #### optimizers ####
        self.optim_drafting = optim.Adam(self.drafting_network.parameters())
        self.optim_revision = optim.Adam(self.revision_network.parameters())
        self.optim_descriminator = optim.Adam(self.descriminator.parameters())
        
        #### losses ####
        self.model_loss_function = LapStyleLoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        #### code for feature extraction ####
        self.vgg_layers = models.vgg19(pretrained=True).features
        for param in self.self.vgg_layers.parameters():
            param.requires_grad = False
        self.capture_layers = [0, 5, 10, 19, 28]
        
    def train(self):
        for epoch in range(n_epochs):
            for i, batch in enumerate(self.dataloader):
                content, style = batch
                content, style = content.to(device), style.to(device)
                
                
                style_down = F.interpolate(
                    style, 
                    [style.shape[2] // 2, style.shape[3] // 2],
                    mode="bilinear", align_corners=False
                    )

                content_down = F.interpolate(
                    content, 
                    [content.shape[2] // 2, content.shape[3] // 2],
                    mode="bilinear", align_corners=False
                    )
                
                ##########################################
                #########    drafting network    #########
                ##########################################
                drafted = self.drafting_network(content_down, style_down)
                
                self.optim_drafting.zero_grad()
                drafting_loss = self.model_loss_function(
                    generated=drafted, 
                    true_content=content_down,
                    true_style=style_down
                    )
                drafting_loss.backward()
                self.optim_drafting.step()
                
                ##########################################
                #########    revision network    #########
                ##########################################     
                draft_no_grad = drafted.detach()
                upsampled_draft = F.interpolate(draft_no_grad, scale_factor=2)
                
                contur = laplacian(content)
                new_pics = self.revision_network(upsampled_draft, contur)
                out_revision = agregate(new_pics, upsampled_draft)

                revision_loss = self.model_loss_function(
                    generated=out_revision, 
                    true_content=content,
                    true_style=style
                    )
                true_preds = self.descriminator(style)
                fake_preds = self.descriminator(out_revision)
                ones = torch.ones_like(pred1, device=self.device)
                zeros = torch.zeros_like(pred2, device=self.device)
                pred_classes = torch.cat([true_preds, fake_preds], dim=0)
                true_classes = torch.cat([ones, zeros], dim=0)
                advers_loss = self.adversarial_loss(pred_classes, true_classes)
                
                self.optim_revision.zero_grad()
                total_revision_loss = revision_loss + self.beta * advers_loss
                total_revision_loss.backward()
                self.optim_drafting.step()
                
                
                ##########################################
                #########      descriminator     #########
                ##########################################
                
                true_preds = self.descriminator(style)
                fake_preds = self.descriminator(out_revision.detach())
                ones = torch.ones_like(pred1, device=self.device)
                zeros = torch.zeros_like(pred2, device=self.device)
                pred_classes = torch.cat([true_preds, fake_preds], dim=0)
                true_classes = torch.cat([ones, zeros], dim=0)
                
                self.descriminator.zero_grad()
                descriminator_loss = - self.adversarial_loss(pred_classes, true_classes)       
                descriminator_loss.backward()
                self.descriminator.step()
                
                
                
                
                
                
                   
        
def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)


def laplacian(x):
    return x - tensor_resample(
        tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]),
        [x.shape[2], x.shape[3]])


def agregate(revised, drafted):
    """
    revised: [batch, 3, 512, 512]
    drafted: [batch, 3, 256, 256]
    """
    h, w = revised.shape[2], revised.shape[3]
    return tensor_resample(drafted, (h, w), mode='bilinear') + revised     