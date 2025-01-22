#!/usr/bin/env python

import argparse
import utils
import logging
import numpy as np
import torch
import sys
import os
from tensorboard_logger import TensorBoardLogger
from torchvision import transforms
import torchvision
import dataset
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from colorama import Style, Fore
from tqdm import tqdm
import torch.optim as optim
from PIL import Image
import io
import datetime
import time
from noise_layers.jpeg import UNetJPEG

torch.backends.cudnn.benchmark = True

def init_prepare(args):
    args.device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    if args.resume is not None:
        run_label = args.resume.split('/')[1]
        checkpoint = torch.load(f'{args.resume}/trained-model-last.pth')
        options = argparse.Namespace(**checkpoint['option'])
        args.__dict__.update(options.__dict__)
        args.epoch += 1
        args.checkpoint = checkpoint
    else:
        args.epoch = 1
        args.step = 1
        args.best_cond = None
        args.best_epoch = -1
        run_name = f'{args.name}_' if args.name is not None else ''
        run_label = f'{run_name}jpegsim_r{args.subset_ratio:.2f}l{args.lr:.5f}b{args.block_size}_{time.strftime("%m%d-%H%M%S")}'

    args.this_run_folder = os.path.join(args.runs, run_label)
    args.image_dir = os.path.join(args.this_run_folder, 'images')

    utils.ensure_dir(args.this_run_folder)
    utils.ensure_dir(args.image_dir)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(args.this_run_folder, f'{run_label}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    logging.info(f'Tensorboard is enabled. Creating logger at {args.this_run_folder}')
    args.tb_logger = TensorBoardLogger(args.this_run_folder)

def compress_image_tensor_batch(image_tensor, quality_factor):
    """
    Compress a batch of image tensors using a given JPEG quality factor.
    
    Args:
        image_tensor (torch.Tensor): The input batch of image tensors with shape (B, C, H, W).
        quality_factor (int): The JPEG quality factor (1-100).
    
    Returns:
        torch.Tensor: The compressed batch of images as tensors.
    """
    compressed_images = []
    
    # Loop over each image in the batch
    for i, img in enumerate(image_tensor):
        # Convert the tensor to a PIL image (expecting the input tensor in range [0, 1])
        transform_to_pil = transforms.ToPILImage()
        pil_image = transform_to_pil(img)
        
        # Compress the image using JPEG with the specified quality factor
        buffer = io.BytesIO()
        q = quality_factor[i].data.item()
        pil_image.save(buffer, format="JPEG", quality=q)

        # Load the compressed image back to a PIL image
        compressed_image = Image.open(buffer)

        # Convert the compressed PIL image back to a tensor
        transform_to_tensor = transforms.ToTensor()
        compressed_image_tensor = transform_to_tensor(compressed_image)

        # Append the compressed image tensor to the list
        compressed_images.append(compressed_image_tensor)

    # Stack all compressed images back into a batch tensor
    return torch.stack(compressed_images)

def save_model(options, model, optimizer, path):
    torch.save({
        'option': options, 
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict()
    }, path)

def load_model(model, optimizer, checkpoint):
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])

def exec_train(args):
    init_prepare(args)

    train_trans = [transforms.ToTensor(),
                   transforms.RandomCrop(args.block_size, pad_if_needed=True, padding_mode='edge')]
    val_trans = [transforms.ToTensor(),
                 transforms.CenterCrop(args.block_size)]
    
    data_feed = dataset.MyDataFeed(args.dataset, train_transform=transforms.Compose(train_trans), val_transform=transforms.Compose(val_trans))

    train_set, val_set = data_feed.load(args.subset_ratio, args.train_ratio, args.block_size, logger = logging.info)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = UNetJPEG().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume is not None:
        logging.info(f'Resume training epoch {args.epoch} to {args.epochs} from {args.resume}/trained-model-last.pth')
        load_model(model, optimizer, args.checkpoint)

    criterion = nn.MSELoss().to(args.device)

    images_to_save = 8
    saved_images_size = (args.block_size, args.block_size)

    # Instantiate the discriminator

    epoch_time = utils.AverageMeter()
    names = ['loss', 'psnr']

    for epoch in range(args.epoch, args.epochs + 1):
        train_log = defaultdict(utils.AverageMeter)
        train_log['lr'].update(optimizer.param_groups[0]["lr"])
        logging.info(f"lr={train_log['lr'].avg:.2e}")
        
        pbar_train = tqdm(train_loader, ncols=100, desc=f'Epoch {epoch}')
        model.train()
        with torch.enable_grad():
            for i, (x, _) in enumerate(pbar_train):
                metrics = {}
                optimizer.zero_grad()

                q = torch.randint(10, 100, (x.shape[0],)).to(args.device)
                x = x.to(args.device)
                j = compress_image_tensor_batch(x, q).to(args.device)
                out_x = model(x, q/100)
                loss = criterion(j, out_x)

                loss.backward()
                optimizer.step()

                metrics['loss'] = loss.detach().data.item()
                metrics['psnr'] = utils.calc_psnr(x, out_x.detach())

                for name, value in metrics.items():
                    train_log[name].update(value)
                if i % 100 == 99:
                    logging.info(f'\rCurr: ' + ' | '.join([f'{n}={train_log[n].avg:.6f}' for n in names]))

        args.tb_logger.save_losses('train_loss', train_log, epoch)

        val_log = defaultdict(utils.AverageMeter)
        val_image_patches = ()
        val_jpeg_patches = ()
        val_encoded_patches = ()
        model.eval()
        with torch.no_grad():
            for x, _ in tqdm(val_loader, ncols=100, desc=f'Epoch {epoch}'):
                metrics = {}
                q = torch.randint(10, 100, (x.shape[0],)).to(args.device)
                x = x.to(args.device)
                j = compress_image_tensor_batch(x, q).to(args.device)
                out_x = model(x, q/100)
                loss = criterion(j, out_x)

                metrics['loss'] = loss.detach().data.item()
                metrics['psnr'] = utils.calc_psnr(j, out_x.detach())

                for name, value in metrics.items():
                    val_log[name].update(value)

                pick = np.random.randint(0, x.shape[0])
                val_image_patches += (F.interpolate(x[pick:pick+1, :, :, :].cpu(), size=saved_images_size),)
                val_encoded_patches += (F.interpolate(out_x[pick:pick+1, :, :, :].cpu(), size=saved_images_size),)
                val_jpeg_patches += (F.interpolate(j[pick:pick+1, :, :, :].cpu(), size=saved_images_size),)

        epoch_time.update(pbar_train.format_dict['elapsed'])
        est_remain = epoch_time.avg * (args.epochs - epoch)
        str_complete = (datetime.datetime.now() + datetime.timedelta(seconds=est_remain)).strftime("%Y-%m-%d %H:%M:%S")
        print(f'Estimate to complete at: {Fore.YELLOW}{Style.BRIGHT}{str_complete}{Style.RESET_ALL}')

        val_image_patches = torch.stack(val_image_patches).squeeze(1)[:images_to_save, :, :, :]
        val_encoded_patches = torch.stack(val_encoded_patches).squeeze(1)[:images_to_save, :, :, :]
        val_jpeg_patches = torch.stack(val_jpeg_patches).squeeze(1)[:images_to_save, :, :, :]
        diff_sim = utils.residual_magnify(val_encoded_patches, val_jpeg_patches)
        stacked_images = torch.cat([val_image_patches, val_jpeg_patches, val_encoded_patches, diff_sim], dim=0)
        torchvision.utils.save_image(stacked_images, f'{args.image_dir}/epoch-{epoch}.png')

        logging.info(f'Test epoch {epoch}/{args.epochs} [{args.best_epoch}]: ' + ' | '.join([f'{n}={val_log[n].avg:.6f}' for n in names]))

        is_best = False
        cond = val_log['loss'].avg
        if args.best_cond is None or cond < args.best_cond:
            args.best_cond = cond
            args.best_epoch = epoch
            logging.info(f"best_cond: psnr = {Fore.CYAN}{Style.BRIGHT}{val_log['psnr'].avg:.6f}{Style.RESET_ALL}, " +
                         f"loss = {Fore.CYAN}{Style.BRIGHT}{val_log['loss'].avg:.6f}{Style.RESET_ALL}")
            is_best = True

        val_log['best_epoch'].update(args.best_epoch)
        args.tb_logger.save_losses('val_loss', val_log, epoch)

        options = {
            'block_size': args.block_size,
            'in_channels': args.in_channels,
            'lr': args.lr,
            'quality': args.quality,
            'epoch': epoch,
            'best_cond': args.best_cond,
            'best_epoch': args.best_epoch,
            'dataset': args.dataset, 
            'subset_ratio': args.subset_ratio,
        }
        save_model(options, model, optimizer, os.path.join(args.this_run_folder, f'trained-model-last.pth'))

        if is_best:
            best_model = os.path.join(args.this_run_folder, f'trained-model.pth')
            logging.info(f'Saving best checkpoint to {best_model}')
            save_model(options, model, optimizer, best_model)

        args.tb_logger.writer.flush()

def exec_predict(args):
    if len(args.images) == 0:
        print('Missing input image')
        return

    args.device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    if args.output is None:
        args.output = f'{args.run_folder}/output'
    utils.ensure_dir(args.output)

    md_name = f'{args.run_folder}/trained-model.pth'
    print(f'load model from {md_name}')
    checkpoint = torch.load(md_name)

    model = UNetJPEG().to(args.device)
    model.load_state_dict(checkpoint['model'])

    img_trans = [transforms.ToTensor()]

    model.eval()
    stats = defaultdict(utils.AverageMeter)
    for im_name in args.images:
        mtx = argparse.Namespace()
        im = Image.open(im_name)
        images = transforms.Compose(img_trans)(im).unsqueeze(0).to(args.device)
        q = torch.randint(10, 100, (images.shape[0],)).to(args.device)
        j = compress_image_tensor_batch(images, q).to(args.device)
        with torch.no_grad():
            out = model(images, q/100)
        mtx.psnr = utils.calc_psnr(j, out)
        print(f'{os.path.basename(im_name)}, q = {q.data.item()}, PSNR = {mtx.psnr:.2f}')

        for m in mtx.__dict__:
            stats[m].update(mtx.__dict__[m])
    
    for m in mtx.__dict__:
        print(f"{m} = {stats[m].avg:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument('--train', dest='dispatch', action='store_const', const=exec_train, default=None, help='train')
    parser.add_argument('--predict', dest='dispatch', action='store_const', const=exec_predict, default=None, help='train')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--quality', '-q', type=int, default=8, help='quality of neural codec')
    parser.add_argument('--block_size', type=int, default=256, help='block size')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--subset_ratio', '-r', type=float, default=0.1, help='subset ratio of dataset')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='ratio of training set')
    parser.add_argument('--dataset', default=None, type=str, help='The directory where the data is stored.')
    parser.add_argument('--name', type=str, default=None, required=False, help='the name of the training')
    parser.add_argument('--in_channels', default=3, type=int, help='input channel size')
    parser.add_argument('--run_folder', '-cp', default=None, type=str, help='The output run folder')
    parser.add_argument('--resume', default=None, type=str, help='The output run folder to resume')
    parser.add_argument('--runs', default='logger', type=str, help='runs folder')
    parser.add_argument('--output', '-o', default=None, type=str, help='output folder')

    args = utils.plug_image_input(parser)

    if args.dispatch is None:
        parser.print_help()
    else:
        args.dispatch(args)
