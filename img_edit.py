#!/usr/bin/env python

from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import numpy as np
import argparse
import utils

def exec_affine(args):
    im_input = Image.open(args.im_in)
    img_rot = im_input.rotate(10, resample=Image.Resampling.BILINEAR).resize((int(im_input.width * 0.95), int(im_input.height * 0.95)))
    rst_image = Image.new("RGB", im_input.size)
    rst_image.paste(img_rot, (10, 0))
    img_rot = rst_image
    img_rot.save(args.im_out)

def exec_jpeg(args):
    im_input = Image.open(args.im_in)
    args.im_out = utils.make_filepath(args.im_out, ext_name='jpg')
    im_input.save(args.im_out, quality=85)

def exec_equalize(args):
    im_input = Image.open(args.im_in)
    img_equ = ImageOps.equalize(im_input)
    img_equ.save(args.im_out)

def exec_blur(args):
    im_input = Image.open(args.im_in)
    img_blur = im_input.filter(ImageFilter.GaussianBlur(radius=2.0))
    img_blur.save(args.im_out)

def exec_median(args):
    im_input = Image.open(args.im_in)
    img_med = im_input.filter(ImageFilter.MedianFilter(size=5))
    img_med.save(args.im_out)

def exec_sharpen(args):
    sharpen_kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
    im_input = Image.open(args.im_in)
    img_sharp = im_input.filter(ImageFilter.Kernel((3, 3), sharpen_kernel.flatten(), scale=1.0))
    img_sharp.save(args.im_out)

def adjust_luminance(img, scale):
    img_light = img.convert("LAB")
    l_channel, a_channel, b_channel = img_light.split()
    l_array = np.array(l_channel, dtype=np.float32)
    l_array *= scale
    l_array = np.clip(l_array, 0, 255)
    adjusted_l_channel = Image.fromarray(l_array.astype(np.uint8))
    img_light = Image.merge("LAB", (adjusted_l_channel, a_channel, b_channel))
    img_light = img_light.convert("RGB")
    return img_light

def exec_lighten(args):
    im_input = Image.open(args.im_in)
    img_light = adjust_luminance(im_input, 1.5)
    img_light.save(args.im_out)

def exec_darken(args):
    im_input = Image.open(args.im_in)
    img_light = adjust_luminance(im_input, 0.5)
    img_light.save(args.im_out)

def exec_copy(args):
    im_input = Image.open(args.im_in)
    BORDER_SIZE = max(im_input.width, im_input.height)//10
    img_center = im_input.crop((BORDER_SIZE, BORDER_SIZE, im_input.width-BORDER_SIZE,im_input.height-BORDER_SIZE))
    img_copy = Image.new('RGB', (im_input.width, im_input.height), (0, 0, 0))
    img_copy.paste(img_center, (BORDER_SIZE, BORDER_SIZE))
    img_copy.save(args.im_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument('--affine', dest='dispatch', action='store_const', const=exec_affine, default=None, help='')
    parser.add_argument('--jpeg', dest='dispatch', action='store_const', const=exec_jpeg, default=None, help='')
    parser.add_argument('--equalize', dest='dispatch', action='store_const', const=exec_equalize, default=None, help='')
    parser.add_argument('--blur', dest='dispatch', action='store_const', const=exec_blur, default=None, help='')
    parser.add_argument('--median', dest='dispatch', action='store_const', const=exec_median, default=None, help='')
    parser.add_argument('--sharpen', dest='dispatch', action='store_const', const=exec_sharpen, default=None, help='')
    parser.add_argument('--lighten', dest='dispatch', action='store_const', const=exec_lighten, default=None, help='')
    parser.add_argument('--darken', dest='dispatch', action='store_const', const=exec_darken, default=None, help='')
    parser.add_argument('--copy', dest='dispatch', action='store_const', const=exec_copy, default=None, help='')
    parser.add_argument('--im_in', '-i', type=str, default=None, help='input image')
    parser.add_argument('--im_out', '-o', type=str, default=None, help='output image')
    args = parser.parse_args()

    if args.dispatch is None:
        parser.print_help()
    else:
        args.dispatch(args)
