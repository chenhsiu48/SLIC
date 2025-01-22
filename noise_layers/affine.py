import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class Affine(nn.Module):
    def __init__(self, degree_range, translate_range, scale_range):
        super(Affine, self).__init__()
        self.degree_range = degree_range
        self.translate_range = translate_range
        self.scale_range = scale_range

    def __str__(self):
        return f'Affine({self.degree_range},{self.translate_range}),({self.scale_range})'

    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
       
        affine = transforms.RandomAffine(degrees=self.degree_range, translate=self.translate_range, scale=self.scale_range, 
                                         interpolation=InterpolationMode.BILINEAR)
        noised_image = affine(noised_image)

        return [noised_image, cover_image]
