import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
    

class DraftingNetwork(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        
        self.img_size = img_size
        self.vgg19 = models.vgg19(pretrained=True).features 
        self.feat_layers = [0, 5, 10, 19]

        self.ada_in = AdaptiveIN()
        self.decode1 = nn.Sequential(
            ResidualBlock(512),
            nn.Conv2d(512, 256, kernel_size=3), 
            nn.ReLU()
        )
        self.decode2 = nn.Sequential(
            ResidualBlock(256),
            nn.Conv2d(256, 128, kernel_size=3), 
            nn.ReLU()
        )
        self.decode3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3), 
            nn.ReLU()
        )
        self.decode4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3), 
        )
        
        
    def forward(self, content, style):
        """
        The function takes in the content and style images and returns the output of the decoder
        
        :param content: The content image
        :param style: The style image
        :return: The output of the last decoder layer.
        """
        
        encoder_states = []
        # Iterating through the encoder layers and applying them to the content and style images.
        
        with torch.no_grad():
            for i in range(len(self.vgg19)):
            
                content = self.vgg19[i](content)
                style = self.vgg19[i](style)
                if i in self.feat_layers:
                    ada_out = self.ada_in(content, style)
                    encoder_states.append(ada_out)

        out = F.interpolate(self.decode1(encoder_states[-1]), size=self.img_size // 4)
        out = F.interpolate(self.decode2(encoder_states[-2] + out), size=self.img_size // 2)
        out = F.interpolate(self.decode3(encoder_states[-3] + out), size=self.img_size )
        return F.interpolate(self.decode4(out), size=self.img_size )


class AdaptiveIN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, content, style):
        """
        content: torch.Tensor[N, C, H, W] - content image
        style: torch.Tensor[N, C, H, W] - style image
        """
        size = content.size()
        content_mean, content_std = self._calculate_mean_std(
            content
        )
        style_mean, style_std = self._calculate_mean_std(
            style
        )
        normalized = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized * style_std.expand(size) + style_mean.expand(size)

    @staticmethod
    def _calculate_mean_std(tensor, eps=1e-5):
        size = tensor.size()
        assert len(size) == 4
        N, C = size[:2]
        tensor_var = tensor.view(N, C, -1).var(dim=2) + eps
        tensor_std = tensor_var.sqrt().view(N, C, 1, 1)
        tensor_mean = tensor.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return tensor_mean, tensor_std
   