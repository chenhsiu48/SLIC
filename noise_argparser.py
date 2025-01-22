import argparse
import re
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.identity import Identity
from noise_layers.gaussian_noise import GaussianNoise
from noise_layers.affine import Affine
from noise_layers.blur import Blur
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.quantization import Quantization
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.jpeg import Jpeg


def parse_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)


def parse_crop(crop_command):
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))

def parse_cropout(cropout_command):
    matches = re.match(r'cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))

def parse_affine(noise_command):
    matches = re.findall(r'\(([\d.+\-]*?),([\d.+\-]*?)\)', noise_command)
    degree_range = (float(matches[0][0]), float(matches[0][1]))
    translate_range = (float(matches[1][0]), float(matches[1][1]))
    scale_range = (float(matches[2][0]), float(matches[2][1]))
    return Affine(degree_range, translate_range, scale_range)

def parse_blur(noise_command):
    matches = re.findall(r'\(([\d.+\-]*?),\(([\d.+\-]*?),([\d.+\-]*?)\)\)', noise_command)
    kernel_size = float(matches[0][0])
    variance_range = (float(matches[0][1]), float(matches[0][2]))
    return Blur(kernel_size, variance_range)

def parse_noise(noise_command):
    matches = re.match(r'noise\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', noise_command)
    (keep_min, keep_max), (var_min, var_max) = parse_pair(matches.groups())
    return GaussianNoise((keep_min, keep_max), (var_min, var_max))

def parse_jpeg(jpeg_command):
    matches = re.match(r'jpeg\((\d+\.*\d*,\d+\.*\d*)\)', jpeg_command)
    ratios = matches.groups()[0].split(',')
    q_min = float(ratios[0])
    q_max = float(ratios[1])
    return Jpeg((q_min, q_max))

def parse_dropout(dropout_command):
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))

def parse_resize(resize_command):
    matches = re.match(r'resize\((\d+\.*\d*,\d+\.*\d*)\)', resize_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))


class NoiseArgParser(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )

    @staticmethod
    def parse_cropout_args(cropout_args):
        pass

    @staticmethod
    def parse_dropout_args(dropout_args):
        pass

    def __call__(self, parser, namespace, values,
                 option_string=None):

        layers = []
        split_commands = values[0].split('+')

        for command in split_commands:
            # remove all whitespace
            command = command.replace(' ', '')
            if command[:len('cropout')] == 'cropout':
                layers.append(parse_cropout(command))
            elif command[:len('crop')] == 'crop':
                layers.append(parse_crop(command))
            elif command[:len('dropout')] == 'dropout':
                layers.append(parse_dropout(command))
            elif command[:len('resize')] == 'resize':
                layers.append(parse_resize(command))
            elif command[:len('noise')] == 'noise':
                layers.append(parse_noise(command))
            elif command[:len('blur')] == 'blur':
                layers.append(parse_blur(command))
            elif command[:len('affine')] == 'affine':
                layers.append(parse_affine(command))
            elif command[:len('jpeg')] == 'jpeg':
                layers.append(parse_jpeg(command))
            elif command[:len('quant')] == 'quant':
                layers.append('QuantizationPlaceholder')
            elif command[:len('identity')] == 'identity':
                # We are adding one Identity() layer in Noiser anyway
                pass
            else:
                raise ValueError('Command not recognized: \n{}'.format(command))
        setattr(namespace, self.dest, layers)
