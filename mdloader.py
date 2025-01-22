import os
import torch
from compressai.zoo.image import model_architectures
from compressai.zoo.pretrained import load_pretrained
import torch.nn as nn
from compressai.zoo import image_models as models
import torch.nn.functional as F

def neural_compressor(model_name, quality, metric="mse", pretrained=False, progress=True, builtin_model=False, model_path=None, **kwargs):
    if builtin_model is False:
        if pretrained:
            if model_path is not None:
                print(f'load snapshot model from local {model_path}')
                state_dict = torch.load(model_path)['state_dict']
                state_dict = load_pretrained(state_dict)
                net = model_architectures[model_name].from_state_dict(state_dict)
                return net

    print(f'load from pretrained {model_name}')
    return models[model_name](quality=quality, metric=metric, pretrained=pretrained, progress=progress, **kwargs)

def load_neural_codec(checkpoint):
    print(f'load from given checkpoint {checkpoint}')
    checkpoint = torch.load(checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict = load_pretrained(state_dict)
    net = model_architectures[checkpoint["model"]].from_state_dict(state_dict)
    return net, (checkpoint['model'], checkpoint['metric'], checkpoint['quality'])

def load_checkpoint(arch: str, no_update: bool, checkpoint_path: str) -> nn.Module:
    print(f'load from given checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict = load_pretrained(state_dict)
    net = model_architectures[checkpoint["model"]].from_state_dict(state_dict)
    return net.eval()
