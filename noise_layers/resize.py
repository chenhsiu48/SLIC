import torch.nn as nn
import torch.nn.functional as F
from noise_layers.crop import random_float

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_range, interpolation_method='bilinear'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method


    def forward(self, noised_and_cover):

        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F.interpolate(
                                    noised_image,
                                    size=(int(noised_image.shape[2] * resize_ratio), int(noised_image.shape[3] * resize_ratio)),
                                    mode=self.interpolation_method, align_corners=True)

        return noised_and_cover
