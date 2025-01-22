#!/usr/bin/env python

import argparse
import utils
import numpy as np
import torch
import os
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
from collections import defaultdict
import torch.nn.functional as F
from PIL import Image
import mdloader
import compressai
from compressai.ops import compute_padding
from mdloader import models
from typing import IO, Dict, NamedTuple, Tuple, Union
import struct

####################################################################################################

metric_ids = {"mse": 0, "ms-ssim": 1}
model_ids = {k: i for i, k in enumerate(models)}

class CodecInfo(NamedTuple):
    codec_header: Tuple
    original_size: Tuple
    original_bitdepth: int
    net: Dict
    device: str

def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4

def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1

def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1

def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    metric = metric_ids[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)

    return model_ids[model_name], code

def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    model_id, code = header
    quality = (code & 0x0F) + 1
    metric = code >> 4

    return (
        inverse_dict(model_ids)[model_id],
        inverse_dict(metric_ids)[metric],
        quality,
    )

def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape

def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt

def pad(x, p=2**6):
    h, w = x.size(2), x.size(3)
    pad, _ = compute_padding(h, w, min_div=p)
    return F.pad(x, pad, mode="replicate")

def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    _, unpad = compute_padding(h, w, out_h=H, out_w=W)
    return F.pad(x, unpad, mode="replicate")

####################################################################################################

def init_prepare(args):
    args.device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    preload_net, (args.model, args.metric, args.quality) = mdloader.load_neural_codec(args.checkpoint)
    print(f'model: {args.model}, metric={args.metric}, quality={args.quality}')
    args.net = preload_net.to(args.device).eval()

    print(f'set entropy coder: {compressai.available_entropy_coders()[0]}')
    compressai.set_entropy_coder(compressai.available_entropy_coders()[0])
    
    utils.ensure_dir(args.out_dir)

def exec_encode(args):
    codec_header_info = get_header(args.model, args.metric, args.quality)
    codec_info = CodecInfo(codec_header_info, None, None, args.net, args.device)

    if args.output is None:
        args.output = utils.make_filepath(args.input, dir_name=args.out_dir, ext_name='bin', tag=f'{args.model}_C')
    
    x = ToTensor()(Image.open(args.input).convert("RGB")).unsqueeze(0)
    x = x.to(args.device)
    bitdepth = 8

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    x = pad(x, p)
    
    with torch.no_grad():
        out = args.net.compress(x)
    
    shape = out["shape"]

    with Path(args.output).open("wb") as f:
        write_uchars(f, codec_header_info)
        # write original image size
        write_uints(f, (h, w))
        # write original bitdepth
        write_uchars(f, (bitdepth,))
        # write shape and number of encoded latents
        write_body(f, shape, out["strings"])

    size = filesize(args.output)
    bpp = float(size) * 8 / (h * w)
    print(f'write to {args.output}, bpp={bpp:.2f}')

def exec_decode(args):
    with Path(args.input).open("rb") as f:
        args.model, args.metric, args.quality = parse_header(read_uchars(f, 2))
        original_size = read_uints(f, 2)
        original_bitdepth = read_uchars(f, 1)[0]
        stream_info = CodecInfo(None, original_size, original_bitdepth, args.net, args.device)
        
        if args.output is None:
            args.output = utils.make_filepath(args.input, dir_name=args.out_dir, ext_name='png')

        strings, shape = read_body(f)
        with torch.no_grad():
            out = args.net.decompress(strings, shape)
        x_hat = crop(out["x_hat"], stream_info.original_size)
        img = ToPILImage()(x_hat.clamp_(0, 1).squeeze())
        print(f'save to {args.output}')
        img.save(args.output)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument('--encode', dest='dispatch', action='store_const', const=exec_encode, default=None, help='encode image')
    parser.add_argument('--decode', dest='dispatch', action='store_const', const=exec_decode, default=None, help='decode image')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument('--checkpoint', '-cp', default='models/lpips-bmshj2018-hyperprior-mse-8-best-1216-050701.pth', type=str, help='model checkpoint')
    parser.add_argument('--out_dir', type=str, required=False, default='output', help='output folder')
    parser.add_argument('--input', '-i', type=str, required=True, help='input file')
    parser.add_argument('--output', '-o', type=str, default=None, help='output file')
    args = parser.parse_args()

    if args.dispatch is None:
        parser.print_help()
    else:
        init_prepare(args)
        args.dispatch(args)
