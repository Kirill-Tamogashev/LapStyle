from typing import Union, Tuple

from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset

class EmptyDirectoryError(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message


class LapStyleDataset(Dataset):
    def __init__(
        self, 
        path_to_content_data: str, 
        path_to_style_data: str, 
        image_size: Tuple[int],
        transform: Union[None, transforms.Compose] = None
        ):
        super().__init__()

        content_dir = os.listdir(path_to_content_data)
        style_dir = os.listdir(path_to_style_data)

        if not content_dir:
            raise EmptyDirectoryError("Content directory is empty")
        if not style_dir:
            raise EmptyDirectoryError("Style directory is empty")

        self.content_images = [
            Image.open(os.path.join(path_to_content_data, img)) 
            for img in content_dir
            ]
        self.style_images = [
            Image.open(os.path.join(path_to_style_data, img)) 
            for img in style_dir
            ]
        if transform is None:
            self.transform  = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform


    def __len__(self):
        return len(self.style_images)
        
    def __getitem__(self, index):
        content = self.content_images[index]
        content = self.transform(content)
        style = self.style_images[0]
        style = self.transform(style)
        
        return {
            "content": content,
            "style": style
        }
        
        