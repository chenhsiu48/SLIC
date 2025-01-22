import torch
import torch.nn as nn
import numpy as np

class GaussianNoise(nn.Module):
    def __init__(self, keep_ratio_range, variance_range):
        super(GaussianNoise, self).__init__()
        self.keep_min = keep_ratio_range[0]
        self.keep_max = keep_ratio_range[1]
        self.var_min = variance_range[0]
        self.var_max = variance_range[1]
        self.is_hidden = False

    def __str__(self):
        return f'GaussianNoise(({self.keep_min},{self.keep_max}),({self.var_min},{self.var_max}))'

    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        mask_percent = np.random.uniform(self.keep_min, self.keep_max)
        variance = np.random.uniform(self.var_min, self.var_max)

        if self.is_hidden:
            noise = cover_image + torch.randn_like(cover_image) * variance * 2
        else:
            noise = cover_image + torch.randn_like(cover_image) * variance + 0.5

        mask = np.random.choice([0.0, 1.0], noised_image.shape[2:], p=[1 - mask_percent, mask_percent])
        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)

        mask_tensor = mask_tensor.expand_as(noised_image)
        
        if self.is_hidden:
            noised_image = noised_image * mask_tensor + (noise * (1-mask_tensor)).clamp(-1, 1)
        else:
            noised_image = noised_image * mask_tensor + (noise * (1-mask_tensor)).clamp(0, 1)
            
        return [noised_image, cover_image]
