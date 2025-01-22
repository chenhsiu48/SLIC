import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class Blur(nn.Module):
    def __init__(self, kernel_size, variance_range):
        super(Blur, self).__init__()
        self.kernel_size = kernel_size
        self.variance_range = variance_range

    def __str__(self):
        return f'Blur({self.kernel_size},{self.variance_range})'

    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
       
        blur_transform = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.variance_range)
        noised_image = blur_transform(noised_image)

        return [noised_image, cover_image]
