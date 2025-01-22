import os
import glob
from argparse import ArgumentParser
import datetime
import numpy as np
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

def read_image_list(fn_list):
    with open(fn_list, 'r') as f:
        res = [l.strip() for l in f.readlines()]
    return res

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def join_path(*dirs):
    if len(dirs) == 0:
        return ''
    path = dirs[0]
    for d in dirs[1:]:
        path = os.path.join(path, d)
    return path

def plug_image_input(parser: ArgumentParser):
    parser.add_argument('--im_path', '-p', nargs='+', type=str, default=None, help='image path')
    parser.add_argument('--im_list', '-l', type=str, default=None, help='image list file')
    parser.add_argument('--im_dir', '-d', type=str, default=None, help='image dir')
    args = parser.parse_args()
    if args.im_list is not None:
        args.images = read_image_list(args.im_list)
    elif args.im_dir is not None:
        args.images = []
        for im_name in glob.glob(join_path(args.im_dir, f'*')):
            if im_name.lower().endswith('.png') or im_name.lower().endswith('.jpg') or im_name.lower().endswith('.jpeg'):
                args.images.append(os.path.abspath(im_name)) 
    elif args.im_path is not None:
        args.images = [os.path.abspath(im_name) for im_name in args.im_path]
    return args

def make_filepath(fpath, dir_name=None, ext_name=None, tag=None):
    if dir_name is None:
        dir_name = os.path.dirname(fpath)
        if dir_name == '':
            dir_name = '.'
    fname = os.path.basename(fpath)
    base, ext = os.path.splitext(fname)
    if ext_name is None:
        ext_name = ext
    elif ext_name != '' and ext_name[0] != '.':
        ext_name = '.' + ext_name
    name = base
    if tag == '':
        name = name.split('-')[0]
    elif tag is not None:
        name = '%s-%s' % (name, tag)
    if ext_name != '':
        name = '%s%s' % (name, ext_name)
    return join_path(dir_name, name)

def mtime_string(file_path):
    last_modification_time = os.path.getmtime(file_path)
    last_modification_time = datetime.datetime.fromtimestamp(last_modification_time)
    return last_modification_time.strftime("%m%d-%H%M%S")

def residual_magnify(src, target):
    import torch
    r = (target - src)
    pos_scale = torch.max(r[r >= 0])
    neg_scale = torch.max(r[r <= 0].abs())
    r[r >= 0] /= pos_scale
    r[r <= 0] /= neg_scale
    r = r / 2 + 0.5
    return r

def calc_psnr(img1, img2):
    import torch
    img1.clamp_(0,1)
    img2.clamp_(0,1)
    img1 = 255 * ((img1 - img1.min()) / (img1.max() - img1.min()))
    img2 = 255 * ((img2 - img2.min()) / (img2.max() - img2.min()))
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))
