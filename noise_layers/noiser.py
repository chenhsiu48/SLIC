import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity
#from noise_layers.jpeg_compression import JpegCompression
from noise_layers.jpeg import Jpeg
from noise_layers.gaussian_noise import GaussianNoise
from noise_layers.quantization import Quantization

class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device, is_hidden = True):
        super(Noiser, self).__init__()
        self.noise_layers = [Identity()]
        for layer in noise_layers:
            if isinstance(layer, Jpeg):
                layer.device = device
                layer.is_hidden = is_hidden
            elif isinstance(layer, GaussianNoise):
                layer.is_hidden = is_hidden
            self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)
        self.embed_eval = False

    def forward(self, encoded_and_cover):
        # set embed_eval to True prevents the identity noiser to be included in the forward pass
        # include identity noiser is only required in the original hidden code
        if self.embed_eval:
            random_noise_layer = np.random.choice(self.noise_layers[1:], 1)[0]
        else:
            random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        return random_noise_layer(encoded_and_cover)
    
    def __str__(self):
        return '+'.join([str(l) for l in self.noise_layers])

    def has_noiser(self):
        return len(self.noise_layers) > 1
