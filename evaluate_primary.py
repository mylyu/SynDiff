#!/usr/bin/env python3
"""Evaluate the primary SynDiff direction: contrast1 -> contrast2."""

import argparse
import csv
import json
import os
import re

import h5py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from backbones.ncsnpp_generator_adagn import NCSNpp
from dataset import CreateDatasetSynthesis
from test import Posterior_Coefficients, get_time_schedule, sample_from_model


def find_latest_epoch(exp_path):
    epochs = []
    for name in os.listdir(exp_path):
        match = re.fullmatch(r'gen_diffusive_2_(\d+)\.pth', name)
        if match:
            epochs.append(int(match.group(1)))
    if not epochs:
        raise FileNotFoundError(f"No gen_diffusive_2 checkpoints found in {exp_path}")
    return max(epochs)


def load_generator(exp_path, epoch, args, device):
    generator = NCSNpp(args).to(device)
    checkpoint = torch.load(
        os.path.join(exp_path, f'gen_diffusive_2_{epoch}.pth'),
        map_location=device,
    )
    state_dict = {}
    for key, value in checkpoint.items():
        state_dict[key[7:] if key.startswith('module.') else key] = value
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['slice_index', 'psnr', 'ssim', 'mae'])
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_chose)
    device = torch.device(f'cuda:{args.gpu_chose}')

    exp_path = os.path.join(args.output_path, args.exp)
    epoch = args.which_epoch if args.which_epoch is not None else find_latest_epoch(exp_path)

    dataset = CreateDatasetSynthesis('test', args.input_path, args.contrast1, args.contrast2)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    generator = load_generator(exp_path, epoch, args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    time_schedule = get_time_schedule(args, device)
    to_range_0_1 = lambda x: (x + 1.) / 2.
    crop = transforms.CenterCrop((args.image_size, args.image_size))

    save_dir = os.path.join(exp_path, 'primary_eval', f'epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    synth_images = np.zeros((len(data_loader), args.image_size, args.image_size), dtype=np.float32)
    source_images = np.zeros_like(synth_images)
    target_images = np.zeros_like(synth_images)

    for index, (source, target) in enumerate(data_loader):
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        x_t = torch.cat((torch.randn_like(target), source), axis=1)

        with torch.no_grad():
            synthetic = sample_from_model(
                pos_coeff,
                generator,
                args.num_timesteps,
                x_t,
                time_schedule,
                args,
            )

        synthetic = crop(torch.clamp(to_range_0_1(synthetic), 0.0, 1.0))
        source_01 = crop(torch.clamp(to_range_0_1(source), 0.0, 1.0))
        target_01 = crop(torch.clamp(to_range_0_1(target), 0.0, 1.0))

        synth_np = np.squeeze(synthetic.cpu().numpy()).astype(np.float32)
        source_np = np.squeeze(source_01.cpu().numpy()).astype(np.float32)
        target_np = np.squeeze(target_01.cpu().numpy()).astype(np.float32)

        synth_images[index] = synth_np
        source_images[index] = source_np
        target_images[index] = target_np

        rows.append({
            'slice_index': index,
            'psnr': float(peak_signal_noise_ratio(target_np, synth_np, data_range=1.0)),
            'ssim': float(structural_similarity(target_np, synth_np, data_range=1.0)),
            'mae': float(np.mean(np.abs(target_np - synth_np))),
        })

        if index % args.preview_stride == 0:
            preview = torch.cat((source_01, synthetic, target_01), axis=-1)
            torchvision.utils.save_image(
                preview,
                os.path.join(save_dir, f'preview_{index:04d}.jpg'),
                normalize=False,
            )

    metrics = {
        'epoch': int(epoch),
        'direction': f'{args.contrast1}->{args.contrast2}',
        'num_slices': len(rows),
        'psnr_mean': float(np.mean([r['psnr'] for r in rows])),
        'psnr_std': float(np.std([r['psnr'] for r in rows])),
        'ssim_mean': float(np.mean([r['ssim'] for r in rows])),
        'ssim_std': float(np.std([r['ssim'] for r in rows])),
        'mae_mean': float(np.mean([r['mae'] for r in rows])),
        'mae_std': float(np.std([r['mae'] for r in rows])),
    }

    write_csv(os.path.join(save_dir, 'per_slice_metrics.csv'), rows)
    with open(os.path.join(save_dir, 'aggregate_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    with h5py.File(os.path.join(save_dir, 'synthetic_0p1T_to_1p5T.mat'), 'w') as f:
        f.create_dataset('source_0p1T', data=source_images)
        f.create_dataset('target_1p5T', data=target_images)
        f.create_dataset('synthetic_1p5T', data=synth_images)

    print(json.dumps(metrics, indent=2))


def build_parser():
    parser = argparse.ArgumentParser('primary SynDiff evaluation')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=2)
    parser.add_argument('--centered', action='store_false', default=True)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
    parser.add_argument('--num_channels_dae', type=int, default=64)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attn_resolutions', default=(16,))
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--resamp_with_conv', action='store_false', default=True)
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--fir', action='store_false', default=True)
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
    parser.add_argument('--skip_rescale', action='store_false', default=True)
    parser.add_argument('--resblock_type', default='biggan')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'])
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'])
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'])
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'])
    parser.add_argument('--fourier_scale', type=float, default=16.)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
    parser.add_argument('--exp', required=True)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr_g', type=float, default=1.6e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--contrast1', type=str, default='0.1T')
    parser.add_argument('--contrast2', type=str, default='1.5T')
    parser.add_argument('--which_epoch', type=int, default=None)
    parser.add_argument('--gpu_chose', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--preview_stride', type=int, default=50)
    return parser


if __name__ == '__main__':
    evaluate(build_parser().parse_args())
