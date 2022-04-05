import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize


from .drafting_network import DraftingNetwork
from .revision_network import RevisionNetwork


def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)


def laplacian(x):
    return x - tensor_resample(
        tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]),
        [x.shape[2], x.shape[3]])


class LapStyle(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.drafting = DraftingNetwork()
        self.revision = RevisionNetwork()
        
    def forward(self, content, style):
        
        content_down = F.interpolate(
            content, 
            [content.shape[2] // 2, content.shape[3] // 2],
            mode="bilinear", align_corners=False
            )
        contur = laplacian(content)
        
        draft = self.drafting(content_down, style)
        upsampled_draft = F.interpolate(draft, scale_factor=2)
        
        revised = self.revision(upsampled_draft, contur)
        return self.agregate(revised, draft)
    
    @staticmethod
    def agregate(revised, drafted):
        """
        revised: [batch, 3, 512, 512]
        drafted: [batch, 3, 256, 256]
        """
        h, w = revised.shape[2], revised.shape[3]
        return tensor_resample(drafted, (h, w), mode='bilinear') + revised
    