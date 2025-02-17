#!/usr/bin/env python

import argparse
import random
import shutil
import time
import dataset
import torch
import torch.nn as nn
import torch.optim as optim
import os
import mdloader
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from colorama import Style, Fore
import datetime
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from tqdm import tqdm
import utils
from compressai.zoo.pretrained import rename_key
import numpy as np
import logging
import sys
from tensorboard_logger import TensorBoardLogger
from collections import defaultdict
from pytorch_msssim import ms_ssim
import math
from compressai.zoo import image_models as models
from lpips import LPIPS
import torch.nn.functional as F
from IQA_pytorch import DISTS, GMSD, NLPD

from noise_argparser import NoiseArgParser
from noise_layers.noiser import Noiser

torch.backends.cudnn.benchmark = True

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        'net': {'type': 'Adam', 'lr': args.learning_rate},
        'aux': {'type': 'Adam', 'lr': args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer['net'], optimizer['aux']

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    #if 'epoch' in state:
    #    if state['epoch'] % 5 == 0:
    #        shutil.copyfile(filename, utils.make_filepath(filename, tag=f"{state['epoch']:04d}"))
    if is_best:
        shutil.copyfile(filename, utils.make_filepath(filename, tag=f'best'))

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).features
        self.layers = layers if layers is not None else [3, 8, 15, 22]  # conv1_2, conv2_2, conv3_3, conv4_3
        self.vgg = nn.Sequential(*list(self.vgg.children())[:max(self.layers) + 1])
        self.vgg.eval()  # Set to evaluation mode

        # Disable gradients since we don't need to update VGG16 weights
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        loss = 0
        x1 = img1
        x2 = img2

        for i, layer in enumerate(self.vgg):
            x1 = layer(x1)
            x2 = layer(x2)
            if i in self.layers:
                loss += torch.nn.functional.mse_loss(x1, x2)  # L1 loss between feature maps
        return loss

ADV_PERCEP_METRICS = ['lpips', 'dists', 'vgg', 'mse', 'ssim', 'nlpd', 'gmsd']

class AdvRateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, adv_percep='lpips', return_type="all"):
        super().__init__()
        self.return_type = return_type
        self.adv_percep = adv_percep
        logging.info(f'adversarial training using {self.adv_percep}')
        if self.adv_percep == 'lpips':
            self.LPIPS = LPIPS(net='vgg')
        elif self.adv_percep == 'dists':
            self.dists = DISTS()
        elif self.adv_percep == 'vgg':
            self.vggloss = VGGPerceptualLoss()
        elif self.adv_percep == 'gmsd':
            self.gmsd = GMSD().eval()
        elif self.adv_percep == 'nlpd':
            self.nlpd = NLPD().eval()

    def forward(self, output, target, recomp, noised_rec, noised_recomp, out):
        TAU = 1.0
        if self.adv_percep == 'lpips':
            out['adv_loss1'] = F.relu(TAU - self.LPIPS(output["x_hat"], recomp['x_hat'], normalize=True).mean(dim=0))
            out['adv_loss2'] = F.relu(TAU - self.LPIPS(noised_rec, noised_recomp['x_hat'], normalize=True).mean(dim=0))
        elif self.adv_percep == 'dists':
            out['adv_loss1'] = F.relu(TAU - self.dists(output["x_hat"], recomp['x_hat'], as_loss=True))
            out['adv_loss2'] = F.relu(TAU - self.dists(noised_rec, noised_recomp['x_hat'], as_loss=True))
        elif self.adv_percep == 'vgg':
            out['adv_loss1'] = F.relu(TAU - self.vggloss(output["x_hat"], recomp['x_hat']))
            out['adv_loss2'] = F.relu(TAU - self.vggloss(noised_rec, noised_recomp['x_hat']))
        elif self.adv_percep == 'mse':
            out['adv_loss1'] = F.relu(TAU - nn.MSELoss()(output["x_hat"], recomp['x_hat']))
            out['adv_loss2'] = F.relu(TAU - nn.MSELoss()(noised_rec, noised_recomp['x_hat']))
        elif self.adv_percep == 'ssim':
            out['adv_loss1'] = ms_ssim(output["x_hat"], recomp['x_hat'], data_range=1)
            out['adv_loss2'] = ms_ssim(noised_rec, noised_recomp['x_hat'], data_range=1)
        elif self.adv_percep == 'gmsd':
            out['adv_loss1'] = F.relu(TAU - self.gmsd(output["x_hat"], recomp['x_hat'], as_loss=True))
            out['adv_loss2'] = F.relu(TAU - self.gmsd(noised_rec, noised_recomp['x_hat'], as_loss=True))
        elif self.adv_percep == 'nlpd':
            out['adv_loss1'] = F.relu(TAU - self.nlpd(output["x_hat"], recomp['x_hat'], as_loss=True))
            out['adv_loss2'] = F.relu(TAU - self.nlpd(noised_rec, noised_recomp['x_hat'], as_loss=True))
        out["adv_loss"] = out['adv_loss1'] + out['adv_loss2']
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
        
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def exec_train(args):

    if args.dataset is None:
        print('Please specify dataset')
        return

    lambda_vals = {
        'mse': [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800],
        'ms-ssim': [2.40, 4.58, 8.73, 16.64, 31.73, 60.50, 115.37, 220.00],
        'lpips': [6, 11.45, 21.83, 41.6, 79.325, 151.25, 288.425, 550.0],
    }
    if args.lmbda is None:
        args.lmbda = lambda_vals[args.metric][args.quality - 1]
    
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size, pad_if_needed=True, padding_mode='edge'), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

    data_feed = dataset.MyDataFeed(args.dataset, train_transform=train_transforms, val_transform=test_transforms)
    train_dataset, test_dataset = data_feed.load(args.subset_ratio, args.train_ratio, logger=logging.info)

    fn_model = utils.make_filepath(f'checkpoint.pth', dir_name=f'{args.this_run_folder}', tag=f'{args.model}-{args.metric}-{args.quality}')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == 'cuda'))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(args.device == 'cuda'))

    net = mdloader.neural_compressor(args.model, args.quality, metric=args.metric, builtin_model=True, pretrained=True)
    net = net.to(args.device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    last_epoch = 1
    best_epoch = -1
    if args.checkpoint:  # load from previous checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        last_epoch = checkpoint['epoch'] + 1
        best_epoch = checkpoint['best_epoch']
        logging.info(f'Load {args.checkpoint}, {last_epoch=}, {best_epoch=}')
        state_dict = checkpoint['state_dict']
        state_dict = {rename_key(k): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint['optimizer'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    criterion = RateDistortionLoss(lmbda=args.lmbda).to(args.device)
    adv_criterion = AdvRateDistortionLoss(adv_percep=args.adv_percep).to(args.device)
    adv_param = [1, 0.2, 0.3, 0.45, 0.6, 0.8, 1, 1.75, 2.5]
    curr_adv_param = adv_param[args.quality]

    names = ['loss', 'adv_loss', f'mse_loss', 'bpp_loss', 'psnr', 'psnr_edit1', 'psnr_edit2', 
             'aux_loss', 'adv_loss1', 'adv_loss2']

    best_loss = float('inf')
    epoch_time = utils.AverageMeter()
    logging.info(f'{args.model=}, {args.quality=}, {args.lmbda=}, {curr_adv_param=}')
    for epoch in range(last_epoch, args.epochs + 1):
        logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}, {aux_optimizer.param_groups[0]["lr"]}, {best_epoch=}')
        if optimizer.param_groups[0]["lr"] < 1e-7:
            logging.info(f'Early exit at {epoch=}, {best_epoch=}')
            break

        train_log = defaultdict(utils.AverageMeter)
        train_log['optimizer_lr'].update(optimizer.param_groups[0]["lr"])
        
        net.train()
        with torch.enable_grad():
            progress = tqdm(train_dataloader, ncols=100, desc=f'Epoch {epoch}/{args.epochs} [{best_epoch}]')
            for i, (x, _) in enumerate(progress):
                x = x.to(args.device)

                optimizer.zero_grad()
                aux_optimizer.zero_grad()

                out_net = net(x)
                out_net['x_hat'].clamp_(0, 1)
                recomp = net(out_net['x_hat'])
                recomp['x_hat'].clamp_(0, 1)
                noised_rec = args.noiser([out_net['x_hat'], out_net['x_hat']])[0]
                noised_recomp = net(noised_rec)
                noised_recomp['x_hat'].clamp_(0, 1)

                out_criterion = criterion(out_net, x)
                out_criterion = adv_criterion(out_net, x, recomp, noised_rec, noised_recomp, out_criterion)
                total_loss = out_criterion['loss'] + curr_adv_param * out_criterion['adv_loss']
                total_loss.backward()
                
                #if args.clip_max_norm > 0:
                #    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)

                optimizer.step()

                out_criterion['aux_loss'] = net.aux_loss()
                out_criterion['aux_loss'].backward()
                aux_optimizer.step()
                
                out_criterion['psnr'] = utils.calc_psnr(x, out_net['x_hat'].detach())
                out_criterion['psnr_edit1'] = utils.calc_psnr(recomp['x_hat'].detach(), out_net['x_hat'].detach())
                out_criterion['psnr_edit2'] = utils.calc_psnr(noised_rec.detach(), noised_recomp['x_hat'].detach())

                for n in names:
                    train_log[n].update(out_criterion[n].detach().data.item())
                if i % 100 == 99:
                    logging.info(f'\rCurr: ' + ' | '.join([f'{n}={train_log[n].avg:.6f}' for n in names]))
                    est_remain = progress.format_dict['elapsed'] / (i + 1) * len(progress) * (args.epochs - epoch)
                    str_complete = (datetime.datetime.now() + datetime.timedelta(seconds=est_remain)).strftime('%m-%d %H:%M')
                    pstate = f'{Fore.YELLOW}{Style.BRIGHT}{str_complete}{Style.RESET_ALL}'
                    progress.set_description(f'Epoch {epoch}/{args.epochs} [{best_epoch}] {pstate}')

        args.tb_logger.save_losses('train_loss', train_log, epoch)

        val_log = defaultdict(utils.AverageMeter)
        num_samples = 8
        net.eval()
        with torch.no_grad():
            val_x_patches = ()
            val_x_hat_patches = ()
            val_xp_hat_patches = ()
            val_xn_hat_patches = ()
            for x, _ in tqdm(test_dataloader, ncols=100, desc=f'Epoch {epoch}'):
                x = x.to(args.device)
                out_net = net(x)
                out_net['x_hat'].clamp_(0, 1)
                recomp = net(out_net['x_hat'])
                recomp['x_hat'].clamp_(0, 1)
                noised_rec = args.noiser([out_net['x_hat'], out_net['x_hat']])[0]
                noised_recomp = net(noised_rec)
                noised_recomp['x_hat'].clamp_(0, 1)

                x_hat = out_net['x_hat']
                pick = np.random.randint(0, x_hat.shape[0])
                val_x_patches += (x[pick:pick+1, :, :, :].cpu(),)
                val_x_hat_patches += (x_hat[pick:pick+1, :, :, :].cpu(),)
                xp_hat = recomp['x_hat']
                val_xp_hat_patches += (xp_hat[pick:pick+1, :, :, :].cpu(),)
                xn_hat = noised_recomp['x_hat']
                val_xn_hat_patches += (xn_hat[pick:pick+1, :, :, :].cpu(),)

                out_criterion = criterion(out_net, x)
                out_criterion = adv_criterion(out_net, x, recomp, noised_rec, noised_recomp, out_criterion)
                out_criterion['aux_loss'] = net.aux_loss()

                total_loss = out_criterion['loss'] + curr_adv_param * out_criterion['adv_loss']
                val_log['total_loss'].update(total_loss.detach().data.item())

                out_criterion['psnr'] = utils.calc_psnr(x, out_net['x_hat'].detach())
                out_criterion['psnr_edit1'] = utils.calc_psnr(recomp['x_hat'].detach(), out_net['x_hat'].detach())
                out_criterion['psnr_edit2'] = utils.calc_psnr(noised_rec.detach(), noised_recomp['x_hat'].detach())

                for n in names:
                    val_log[n].update(out_criterion[n].data.item())

            pick = np.random.randint(0, len(val_x_patches) - 8)
            val_x_patches = torch.stack(val_x_patches).squeeze(1)[pick:pick+num_samples, :, :, :]
            val_x_hat_patches = torch.stack(val_x_hat_patches).squeeze(1)[pick:pick+num_samples, :, :, :]
            val_xp_hat_patches = torch.stack(val_xp_hat_patches).squeeze(1)[pick:pick+num_samples, :, :, :]
            val_xn_hat_patches = torch.stack(val_xn_hat_patches).squeeze(1)[pick:pick+num_samples, :, :, :]
            diff_encode = utils.residual_magnify(val_x_patches, val_x_hat_patches)
            diff_recomp = utils.residual_magnify(val_x_hat_patches, val_xp_hat_patches)
            stacked_images = torch.cat([val_x_patches, val_x_hat_patches, diff_encode, val_xp_hat_patches, diff_recomp, val_xn_hat_patches], dim=0)
            torchvision.utils.save_image(stacked_images, f'{args.image_dir}/epoch-{epoch:04d}.png')
        
        args.tb_logger.save_losses('val_loss', val_log, epoch)
        
        logging.info(f'Test epoch {epoch}: ' + ' | '.join([f'{n}={val_log[n].avg:.6f}' for n in names]))
        loss = val_log['total_loss'].avg
        lr_scheduler.step(loss)

        epoch_time.update(progress.format_dict['elapsed'])
        est_remain = epoch_time.avg * (args.epochs - epoch)
        str_complete = (datetime.datetime.now() + datetime.timedelta(seconds=est_remain)).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'Estimate to complete at: {Fore.YELLOW}{Style.BRIGHT}{str_complete}{Style.RESET_ALL}')

        is_best = loss < best_loss
        if is_best:
            best_loss = loss
            best_epoch = epoch

        net.update()
        save_checkpoint({
                'model': args.model, 'metric': args.metric, 'quality': args.quality, 'ratio': args.subset_ratio,
                'epoch': epoch, 'best_epoch': best_epoch,
                'state_dict': net.state_dict(),
                'loss': loss, 'optimizer': optimizer.state_dict(), 'aux_optimizer': aux_optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),
            },
            is_best, fn_model
        )
        args.tb_logger.writer.flush()
    
    fn_best = utils.make_filepath(fn_model, tag=f'best')
    mtime = utils.mtime_string(fn_best)
    shutil.copyfile(fn_best, utils.make_filepath(fn_best, dir_name=args.out_dir, tag=mtime))
    os.rename(fn_best, utils.make_filepath(fn_best, tag=mtime))

def init_prepare(args):
    args.device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if args.resume is None:
        name = f'{args.name}_' if args.name else ''
        run_label = f'{name}{args.model}_r{args.subset_ratio:.2f}_{args.metric}_q{args.quality}_{time.strftime("%m%d-%H%M%S")}'
        args.checkpoint = None
    else:
        run_label = args.resume.split('/')[-1]
        args.checkpoint = utils.make_filepath(f'checkpoint.pth', dir_name=args.resume, tag=f'{args.model}-{args.metric}-{args.quality}')
        
    args.this_run_folder = os.path.join(args.runs, run_label)
    args.image_dir = os.path.join(args.this_run_folder, 'images')

    utils.ensure_dir(args.out_dir)
    utils.ensure_dir(args.this_run_folder)
    utils.ensure_dir(args.image_dir)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(args.this_run_folder, f'{run_label}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    logging.info(f'Tensorboard is enabled. Creating logger at {args.this_run_folder}')
    args.tb_logger = TensorBoardLogger(args.this_run_folder)

    args.noiser = Noiser(args.noise, args.device, is_hidden=False)
    args.embed_eval = True
    logging.info(f'Noise layers: {args.noiser}')

if __name__ == '__main__':
    model_names = list(models.keys())
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument('--train', dest='dispatch', action='store_const', const=exec_train, default=None, help='')
    parser.add_argument('--model', '-m', default='bmshj2018-hyperprior', choices=model_names, help='Model architecture (default: %(default)s)')
    parser.add_argument('--dataset', '-d', type=str, default='data/COCO', required=False, help='Training dataset')
    parser.add_argument('--name', type=str, default=None, required=False, help='name the training')
    parser.add_argument('--subset_ratio', '-r', type=float, default=0.2, help='subset ratio of dataset')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='ratio of training set')
    parser.add_argument('--quality', '-q', type=int, default=8, help='quality of neural codec')
    parser.add_argument('--epochs', '-e', default=200, type=int, help='Number of epochs (default: %(default)s)')
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float, help='Learning rate (default: %(default)s)')
    parser.add_argument('--aux_learning_rate', '-aux_lr', type=float, default=1e-3, help='Auxiliary loss learning rate (default: %(default)s)')
    parser.add_argument('--num-workers', '-n', type=int, default=4, help='Dataloaders threads (default: %(default)s)')
    parser.add_argument('--lambda', dest='lmbda', type=float, default=None, help='Bit-rate distortion parameter (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: %(default)s)')
    parser.add_argument('--patch-size', type=int, nargs=2, default=(256, 256), help='Size of the patches to be cropped (default: %(default)s)')
    parser.add_argument('--metric', choices=['mse', 'ms-ssim'], default='mse', help='metric trained against (default: %(default)s)')
    parser.add_argument('--adv_percep', choices=ADV_PERCEP_METRICS, default='lpips', help='perceptual loss for adv (default: %(default)s)')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument('--seed', type=int, help='Set random seed for reproducibility')
    parser.add_argument('--clip_max_norm', default=1.0, type=float, help='gradient clipping max norm (default: %(default)s')
    parser.add_argument('--resume', type=str, help='resume from a run folder')
    parser.add_argument('--runs', default='logger', type=str, help='runs folder')
    parser.add_argument('--out_dir', default='models', type=str, help='model output folder')
    parser.add_argument('--noise', nargs='*', action=NoiseArgParser, help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")
    args = parser.parse_args()

    if args.dispatch is None:
        parser.print_help()
    else:
        init_prepare(args)
        args.dispatch(args)
