#!/usr/bin/env python3
"""Train SynDiff non-diffusive generators only (CycleGAN-style, no diffusion).

Tests whether the ResnetGenerator (6 blocks) can learn translation at all.
Uses the same cycle discriminators and cycle-consistency loss as SynDiff,
but with lambda_cycle=10.0 (matching baseline CycleGAN).

Usage:
    python3 experiments/train_nondiff_only.py --batch_size 4 --num_epoch 10 --gpu 0
"""
import argparse, os, sys, copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
import backbones.generator_resnet
from dataset import CreateDatasetSynthesis
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def compute_nrmse(pred, target):
    d = pred.astype(np.float64) - target.astype(np.float64)
    n = np.linalg.norm(target.astype(np.float64))
    return float(np.linalg.norm(d)/n) if n > 1e-10 else 0.0

def compute_ssim_official(pred, target):
    vals = []
    for p, t in zip(pred, target):
        dr = t.max() - t.min()
        vals.append(1.0 if dr<1e-10 else ssim(t, p, data_range=dr))
    return float(np.mean(vals))

def train_nondiff(rank, gpu, args):
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    device = torch.device(f'cuda:{gpu}')
    B = args.batch_size
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # Dataset
    dataset = CreateDatasetSynthesis('train', args.input_path, args.contrast1, args.contrast2)
    dataset_val = CreateDatasetSynthesis('val', args.input_path, args.contrast1, args.contrast2)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False, num_workers=2,
                                               pin_memory=True, sampler=train_sampler, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, num_replicas=args.world_size, rank=rank)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2,
                                                   pin_memory=True, sampler=val_sampler, drop_last=True)

    # Only non-diffusive generators + cycle discriminators
    gen_1to2 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[gpu])
    gen_2to1 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[gpu])
    disc_1 = backbones.generator_resnet.define_D(gpu_ids=[gpu])
    disc_2 = backbones.generator_resnet.define_D(gpu_ids=[gpu])

    broadcast_params(gen_1to2.parameters()); broadcast_params(gen_2to1.parameters())
    broadcast_params(disc_1.parameters()); broadcast_params(disc_2.parameters())

    # DDP wrap
    gen_1to2 = nn.parallel.DistributedDataParallel(gen_1to2, device_ids=[gpu], find_unused_parameters=True)
    gen_2to1 = nn.parallel.DistributedDataParallel(gen_2to1, device_ids=[gpu], find_unused_parameters=True)
    disc_1 = nn.parallel.DistributedDataParallel(disc_1, device_ids=[gpu], find_unused_parameters=True)
    disc_2 = nn.parallel.DistributedDataParallel(disc_2, device_ids=[gpu], find_unused_parameters=True)

    opt_gen = optim.Adam(list(gen_1to2.parameters()) + list(gen_2to1.parameters()),
                         lr=args.lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(list(disc_1.parameters()) + list(disc_2.parameters()),
                          lr=args.lr, betas=(0.5, 0.999))

    exp_path = os.path.join(args.output_path, args.exp)
    if rank == 0:
        os.makedirs(exp_path, exist_ok=True)

    for epoch in range(1, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)
        for iteration, (x1, x2) in enumerate(data_loader):
            real_1 = x1.to(device); real_2 = x2.to(device)

            # --- D phase ---
            disc_1.zero_grad(); disc_2.zero_grad()
            # D real
            D1_real = disc_1(real_1).view(-1); D2_real = disc_2(real_2).view(-1)
            loss_D_real = (nn.functional.softplus(-D1_real.float()).mean() +
                           nn.functional.softplus(-D2_real.float()).mean())
            # D fake
            with torch.no_grad():
                fake_1 = gen_2to1(real_2); fake_2 = gen_1to2(real_1)
            D1_fake = disc_1(fake_1.detach()).view(-1); D2_fake = disc_2(fake_2.detach()).view(-1)
            loss_D_fake = (nn.functional.softplus(D1_fake.float()).mean() +
                           nn.functional.softplus(D2_fake.float()).mean())
            (loss_D_real + loss_D_fake).backward()
            opt_disc.step()

            # --- G phase ---
            gen_1to2.zero_grad(); gen_2to1.zero_grad()
            fake_1 = gen_2to1(real_2); fake_2 = gen_1to2(real_1)
            cycle_1 = gen_2to1(fake_2); cycle_2 = gen_1to2(fake_1)

            # GAN loss
            D1_fake = disc_1(fake_1).view(-1); D2_fake = disc_2(fake_2).view(-1)
            loss_G_adv = (nn.functional.softplus(-D1_fake.float()).mean() +
                          nn.functional.softplus(-D2_fake.float()).mean())
            # Cycle loss (lambda_cycle=10.0 matching baseline CycleGAN)
            loss_cycle = (nn.functional.l1_loss(cycle_1, real_1) +
                          nn.functional.l1_loss(cycle_2, real_2)) * 10.0
            (loss_G_adv + loss_cycle).backward()
            opt_gen.step()

            if iteration % 100 == 0 and rank == 0:
                print(f'epoch {epoch} iter {iteration}: G_adv={loss_G_adv.item():.4f} '
                      f'cycle={loss_cycle.item():.4f} D={loss_D_real.item()+loss_D_fake.item():.4f}')

        # === Validation (paired) ===
        gen1 = gen_2to1.module if hasattr(gen_2to1, 'module') else gen_2to1
        gen2 = gen_1to2.module if hasattr(gen_1to2, 'module') else gen_1to2
        nrmse_d1, ssim_d1, nrmse_d2, ssim_d2 = [], [], [], []
        for x_val, y_val in data_loader_val:
            r1, r2 = x_val.to(device), y_val.to(device)
            with torch.no_grad():
                p1 = gen1(r2); p2 = gen2(r1)
            p1n = to_range_0_1(p1).squeeze().cpu().numpy()
            p2n = to_range_0_1(p2).squeeze().cpu().numpy()
            r1n = to_range_0_1(r1).squeeze().cpu().numpy()
            r2n = to_range_0_1(r2).squeeze().cpu().numpy()
            nrmse_d1.append(compute_nrmse(p1n, r1n))
            ssim_d1.append(compute_ssim_official(p1n[np.newaxis], r1n[np.newaxis]))
            nrmse_d2.append(compute_nrmse(p2n, r2n))
            ssim_d2.append(compute_ssim_official(p2n[np.newaxis], r2n[np.newaxis]))

        if rank == 0:
            print(f'VAL epoch {epoch}: nrmse_d1={np.mean(nrmse_d1):.4f} ssim_d1={np.mean(ssim_d1):.4f} '
                  f'nrmse_d2={np.mean(nrmse_d2):.4f} ssim_d2={np.mean(ssim_d2):.4f}')
            # Save checkpoints
            torch.save(gen1.state_dict(), f'{exp_path}/gen_nondiff_2to1_{epoch}.pth')
            torch.save(gen2.state_dict(), f'{exp_path}/gen_nondiff_1to2_{epoch}.pth')

    dist.barrier()
    dist.destroy_process_group()

def init_processes(rank, size, fn, args, local_rank):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, local_rank, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='/data0/syndiff_data/1.5T_to_7T')
    parser.add_argument('--output_path', default='/NAS_writeable/SynDiff/checkpoints')
    parser.add_argument('--exp', default='nondiff_cyclegan')
    parser.add_argument('--contrast1', default='1.5T')
    parser.add_argument('--contrast2', default='7T')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_process_per_node', type=int, default=2)
    parser.add_argument('--master_address', default='127.0.0.1')
    parser.add_argument('--port_num', default='6160')
    args = parser.parse_args()
    args.world_size = args.num_process_per_node

    processes = []
    for rank in range(args.num_process_per_node):
        pa = copy.deepcopy(args)
        p = Process(target=init_processes, args=(rank, args.world_size, train_nondiff, pa, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
