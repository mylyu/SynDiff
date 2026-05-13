#!/usr/bin/env python3
"""Experiment A: SynDiff + baseline npz data.

Same SynDiff model and training loop as train.py, but uses baseline's
CachedUnpairedDataset via SynDiffNPZAdapter instead of our .mat data.

Usage:
    python3 experiments/run_exp_a.py --preprocessed_dir /path/to/npz_cache \
        --src_field 1.5T --tgt_field 7T [same SynDiff args as train.py]
"""
import argparse, copy, os, sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process
import torchvision
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add code dir and experiments dir to path
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'experiments'))

from exp_a_adapter import SynDiffNPZAdapter

# ---------- Reuse train.py's helper functions ----------
# (Copied from train.py — identical logic, just different dataset creation)

def compute_nrmse(pred, target):
    diff = pred.astype(np.float64) - target.astype(np.float64)
    norm_t = np.linalg.norm(target.astype(np.float64))
    return float(np.linalg.norm(diff) / norm_t) if norm_t > 1e-10 else 0.0

def _as_2d_slices(array):
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError('SSIM expects at least 2 image dimensions')
    if array.ndim == 2:
        return array.reshape((1,) + array.shape)
    return array.reshape((-1,) + array.shape[-2:])

def compute_ssim_slices(pred, target, data_range=1.0):
    pred_slices = _as_2d_slices(pred)
    target_slices = _as_2d_slices(target)
    values = [ssim(t, p, data_range=data_range) for p, t in zip(pred_slices, target_slices)]
    return float(np.mean(values)) if values else np.nan

def compute_ssim_official(pred, target):
    pred_slices = _as_2d_slices(pred)
    target_slices = _as_2d_slices(target)
    values = []
    for p, t in zip(pred_slices, target_slices):
        dr = t.max() - t.min()
        values.append(1.0 if dr < 1e-10 else ssim(t, p, data_range=dr))
    return float(np.mean(values)) if values else np.nan

def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def rank_log(args, rank, message):
    if getattr(args, 'debug_sync_trace', False):
        print('[rank{}] {}'.format(rank, message), flush=True)

# Diffusion / posterior functions — identical to train.py
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    return 1. - torch.exp(2. * log_mean_coeff)

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64) / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min, beta_max = args.beta_min, args.beta_max
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64) / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device).float()
    return betas**0.5, torch.sqrt(1 - betas), betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)

def q_sample(coeff, x_start, t, *, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    return (extract(coeff.a_s_cum, t, x_start.shape) * x_start +
            extract(coeff.sigmas_cum, t, x_start.shape) * noise)

def q_sample_pairs(coeff, x_start, t):
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = (extract(coeff.a_s, t+1, x_start.shape) * x_t +
                    extract(coeff.sigmas, t+1, x_start.shape) * noise)
    return x_t, x_t_plus_one

class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t)
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise
    return p_sample(x_0, x_t, t)

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init[:, [0], :]
    source = x_init[:, [1], :]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(torch.cat((x, source), axis=1), t, latent_z)
            x_new = sample_posterior(coefficients, x_0[:, [0], :], x, t)
            x = x_new.detach()
    return x

# ---------- Training function (identical to train.py except dataset creation) ----------

def train_syndiff(rank, gpu, args):
    from backbones.discriminator import Discriminator_small, Discriminator_large
    from backbones.ncsnpp_generator_adagn import NCSNpp
    import backbones.generator_resnet
    from utils.EMA import EMA
    import shutil

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size
    nz = args.nz

    # === ONLY CHANGE FROM train.py: Use npz adapter instead of .mat data ===
    dataset = SynDiffNPZAdapter(
        preprocessed_dir=args.preprocessed_dir,
        split="retro_train",
        modality="T1W",
        src_field=args.src_field,
        tgt_field=args.tgt_field,
        crop_size=(args.image_size, args.image_size),
    )
    dataset_val = SynDiffNPZAdapter(
        preprocessed_dir=args.preprocessed_dir,
        split="retro_train",
        modality="T1W",
        src_field=args.src_field,
        tgt_field=args.tgt_field,
        crop_size=(args.image_size, args.image_size),
    )
    # ====================================================================

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=args.world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True, sampler=train_sampler, drop_last=True,
        persistent_workers=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_val, num_replicas=args.world_size, rank=rank)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True, sampler=val_sampler, drop_last=True,
        persistent_workers=True)

    val_l1_loss = np.zeros([2, args.num_epoch + 1, len(data_loader_val)])
    val_psnr_values = np.zeros([2, args.num_epoch + 1, len(data_loader_val)])
    val_ssim_values = np.zeros([2, args.num_epoch + 1, len(data_loader_val)])
    val_raw_psnr = np.zeros([2, args.num_epoch + 1, len(data_loader_val)])
    val_raw_ssim = np.zeros([2, args.num_epoch + 1, len(data_loader_val)])

    print(f'train data size:{len(data_loader)}')
    print(f'val data size:{len(data_loader_val)}')
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # Models
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)
    args.num_channels = 1
    gen_non_diffusive_1to2 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[gpu])
    gen_non_diffusive_2to1 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[gpu])

    disc_diffusive_1 = Discriminator_large(nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim,
                                             act=nn.LeakyReLU(0.2)).to(device)
    disc_diffusive_2 = Discriminator_large(nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim,
                                             act=nn.LeakyReLU(0.2)).to(device)
    disc_non_diffusive_cycle1 = backbones.generator_resnet.define_D(gpu_ids=[gpu])
    disc_non_diffusive_cycle2 = backbones.generator_resnet.define_D(gpu_ids=[gpu])

    broadcast_params(gen_diffusive_1.parameters())
    broadcast_params(gen_diffusive_2.parameters())
    broadcast_params(gen_non_diffusive_1to2.parameters())
    broadcast_params(gen_non_diffusive_2to1.parameters())
    broadcast_params(disc_diffusive_1.parameters())
    broadcast_params(disc_diffusive_2.parameters())
    broadcast_params(disc_non_diffusive_cycle1.parameters())
    broadcast_params(disc_non_diffusive_cycle2.parameters())

    # Optimizers
    optimizer_disc_diffusive_1 = optim.Adam(disc_diffusive_1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_disc_diffusive_2 = optim.Adam(disc_diffusive_2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_gen_diffusive_1 = optim.Adam(gen_diffusive_1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_gen_diffusive_2 = optim.Adam(gen_diffusive_2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_gen_non_diffusive_1to2 = optim.Adam(gen_non_diffusive_1to2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_gen_non_diffusive_2to1 = optim.Adam(gen_non_diffusive_2to1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_disc_non_diffusive_cycle1 = optim.Adam(disc_non_diffusive_cycle1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_disc_non_diffusive_cycle2 = optim.Adam(disc_non_diffusive_cycle2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=args.ema_decay)
        optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=args.ema_decay)
        optimizer_gen_non_diffusive_1to2 = EMA(optimizer_gen_non_diffusive_1to2, ema_decay=args.ema_decay)
        optimizer_gen_non_diffusive_2to1 = EMA(optimizer_gen_non_diffusive_2to1, ema_decay=args.ema_decay)

    # Schedulers
    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_1to2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_1to2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_2to1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_2to1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_2, args.num_epoch, eta_min=1e-5)
    scheduler_disc_non_diffusive_cycle1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_non_diffusive_cycle2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle2, args.num_epoch, eta_min=1e-5)

    # DDP
    gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, device_ids=[gpu], find_unused_parameters=True)
    gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, device_ids=[gpu], find_unused_parameters=True)
    gen_non_diffusive_1to2 = nn.parallel.DistributedDataParallel(gen_non_diffusive_1to2, device_ids=[gpu], find_unused_parameters=True)
    gen_non_diffusive_2to1 = nn.parallel.DistributedDataParallel(gen_non_diffusive_2to1, device_ids=[gpu], find_unused_parameters=True)
    disc_diffusive_1 = nn.parallel.DistributedDataParallel(disc_diffusive_1, device_ids=[gpu], find_unused_parameters=True)
    disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, device_ids=[gpu], find_unused_parameters=True)
    disc_non_diffusive_cycle1 = nn.parallel.DistributedDataParallel(disc_non_diffusive_cycle1, device_ids=[gpu], find_unused_parameters=True)
    disc_non_diffusive_cycle2 = nn.parallel.DistributedDataParallel(disc_non_diffusive_cycle2, device_ids=[gpu], find_unused_parameters=True)

    exp_path = os.path.join(args.output_path, args.exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree(os.path.join(CODE_DIR, 'backbones'), os.path.join(exp_path, 'backbones'))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        # ... (resume logic omitted for brevity — same as train.py)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
        gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])
        global_step = checkpoint['global_step']
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    # === TRAINING LOOP (identical to train.py) ===
    for epoch in range(init_epoch + 1, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)
        for iteration, (x1, x2) in enumerate(data_loader):
            # --- D phase (real) ---
            for p in disc_diffusive_1.parameters(): p.requires_grad = True
            for p in disc_diffusive_2.parameters(): p.requires_grad = True
            for p in disc_non_diffusive_cycle1.parameters(): p.requires_grad = True
            for p in disc_non_diffusive_cycle2.parameters(): p.requires_grad = True

            disc_diffusive_1.zero_grad(); disc_diffusive_2.zero_grad()
            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)
            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
            x1_t.requires_grad = True; x2_t.requires_grad = True

            D1_real = disc_diffusive_1(x1_t, t1, x1_tp1.detach()).view(-1)
            D2_real = disc_diffusive_2(x2_t, t2, x2_tp1.detach()).view(-1)
            errD_real = (torch.nn.functional.softplus(-D1_real.float()).mean() +
                         torch.nn.functional.softplus(-D2_real.float()).mean())
            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None or global_step % args.lazy_reg == 0:
                grad1_real = torch.autograd.grad(outputs=D1_real.float().sum(), inputs=x1_t, create_graph=True)[0]
                grad2_real = torch.autograd.grad(outputs=D2_real.float().sum(), inputs=x2_t, create_graph=True)[0]
                grad_penalty = args.r1_gamma / 2 * (grad1_real.pow(2).mean() + grad2_real.pow(2).mean())
                grad_penalty.backward()

            # --- D phase (fake) ---
            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)
            with (gen_diffusive_1.no_sync(), gen_diffusive_2.no_sync(),
                  gen_non_diffusive_1to2.no_sync(), gen_non_diffusive_2to1.no_sync()):
                x1_0_predict = gen_non_diffusive_2to1(real_data2)
                x2_0_predict = gen_non_diffusive_1to2(real_data1)
                x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict), axis=1), t1, latent_z1)
                x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2)
                x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1)
                x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], :], x2_tp1, t2)
                output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
                output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)
                errD_fake = (torch.nn.functional.softplus(output1.float()).mean() +
                             torch.nn.functional.softplus(output2.float()).mean())
                errD_fake.backward()

            optimizer_disc_diffusive_1.step(); optimizer_disc_diffusive_2.step()

            # --- D cycle ---
            disc_non_diffusive_cycle1.zero_grad(); disc_non_diffusive_cycle2.zero_grad()
            D_cycle1_real = disc_non_diffusive_cycle1(real_data1).view(-1)
            D_cycle2_real = disc_non_diffusive_cycle2(real_data2).view(-1)
            errD_cycle_real = (torch.nn.functional.softplus(-D_cycle1_real.float()).mean() +
                               torch.nn.functional.softplus(-D_cycle2_real.float()).mean())
            errD_cycle_real.backward(retain_graph=True)

            x1_0_predict_cycle = gen_non_diffusive_2to1(real_data2)
            x2_0_predict_cycle = gen_non_diffusive_1to2(real_data1)
            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict_cycle).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict_cycle).view(-1)
            errD_cycle_fake = (torch.nn.functional.softplus(D_cycle1_fake.float()).mean() +
                               torch.nn.functional.softplus(D_cycle2_fake.float()).mean())
            errD_cycle_fake.backward()
            optimizer_disc_non_diffusive_cycle1.step(); optimizer_disc_non_diffusive_cycle2.step()

            # --- G phase ---
            for p in disc_diffusive_1.parameters(): p.requires_grad = False
            for p in disc_diffusive_2.parameters(): p.requires_grad = False
            for p in disc_non_diffusive_cycle1.parameters(): p.requires_grad = False
            for p in disc_non_diffusive_cycle2.parameters(): p.requires_grad = False
            gen_diffusive_1.zero_grad(); gen_diffusive_2.zero_grad()
            gen_non_diffusive_1to2.zero_grad(); gen_non_diffusive_2to1.zero_grad()

            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)

            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict_cycle = gen_non_diffusive_1to2(x1_0_predict)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)
            x1_0_predict_cycle = gen_non_diffusive_2to1(x2_0_predict)
            x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict), axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2)
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], :], x2_tp1, t2)
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)
            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1)

            errG_adv = (torch.nn.functional.softplus(-output1.float()).mean() +
                        torch.nn.functional.softplus(-output2.float()).mean())
            errG_cycle_adv = (torch.nn.functional.softplus(-D_cycle1_fake.float()).mean() +
                              torch.nn.functional.softplus(-D_cycle2_fake.float()).mean())
            errG_L1 = (torch.nn.functional.l1_loss(x1_0_predict_diff[:, [0], :], real_data1) +
                       torch.nn.functional.l1_loss(x2_0_predict_diff[:, [0], :], real_data2))
            errG_cycle = (torch.nn.functional.l1_loss(x1_0_predict_cycle, real_data1) +
                          torch.nn.functional.l1_loss(x2_0_predict_cycle, real_data2))
            errG = args.lambda_l1_loss * errG_cycle + errG_adv + errG_cycle_adv + args.lambda_l1_loss * errG_L1
            errG.backward()

            optimizer_gen_diffusive_1.step(); optimizer_gen_diffusive_2.step()
            optimizer_gen_non_diffusive_1to2.step(); optimizer_gen_non_diffusive_2to1.step()
            global_step += 1

            if iteration % 100 == 0 and rank == 0:
                print(f'epoch {epoch} iteration{iteration}, G-Cycle: {errG_cycle.item():.4f}, '
                      f'G-L1: {errG_L1.item():.4f}, G-Adv: {errG_adv.item():.4f}, '
                      f'G-cycle-Adv: {errG_cycle_adv.item():.4f}, G-Sum: {errG.item():.4f}, '
                      f'D Loss: {errD_real.item() + errD_fake.item():.4f}')

        # === LR decay ===
        if not args.no_lr_decay:
            scheduler_gen_diffusive_1.step(); scheduler_gen_diffusive_2.step()
            scheduler_gen_non_diffusive_1to2.step(); scheduler_gen_non_diffusive_2to1.step()
            scheduler_disc_diffusive_1.step(); scheduler_disc_diffusive_2.step()
            scheduler_disc_non_diffusive_cycle1.step(); scheduler_disc_non_diffusive_cycle2.step()

        # === Epoch-end: sample, save, validate ===
        gen1 = gen_diffusive_1.module if hasattr(gen_diffusive_1, 'module') else gen_diffusive_1
        gen2 = gen_diffusive_2.module if hasattr(gen_diffusive_2, 'module') else gen_diffusive_2
        nd1to2 = gen_non_diffusive_1to2.module if hasattr(gen_non_diffusive_1to2, 'module') else gen_non_diffusive_1to2
        nd2to1 = gen_non_diffusive_2to1.module if hasattr(gen_non_diffusive_2to1, 'module') else gen_non_diffusive_2to1

        if rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(x1_pos_sample, os.path.join(exp_path, f'xpos1_epoch_{epoch}.png'), normalize=True)
                torchvision.utils.save_image(x2_pos_sample, os.path.join(exp_path, f'xpos2_epoch_{epoch}.png'), normalize=True)
            # content save
            if args.save_content and epoch % args.save_content_every == 0:
                print('Saving content.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                           'gen_diffusive_1_dict': gen_diffusive_1.state_dict(),
                           'gen_diffusive_2_dict': gen_diffusive_2.state_dict(),
                           'gen_non_diffusive_1to2_dict': gen_non_diffusive_1to2.state_dict(),
                           'gen_non_diffusive_2to1_dict': gen_non_diffusive_2to1.state_dict()}
                torch.save(content, os.path.join(exp_path, 'content.pth'))
                torch.cuda.empty_cache()

        dist.barrier()

        # === Validation ===
        for iteration, (x_val, y_val) in enumerate(data_loader_val):
            real_data = x_val.to(device, non_blocking=True)
            source_data = y_val.to(device, non_blocking=True)
            x1_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
            fake_sample1 = sample_from_model(pos_coeff, gen1, args.num_timesteps, x1_t, T, args)
            fake_sample1 = to_range_0_1(fake_sample1)
            real_data_n = to_range_0_1(real_data)
            fake_raw = fake_sample1.cpu().numpy(); real_raw = real_data_n.cpu().numpy()
            fake_sample1 = fake_sample1 / fake_sample1.max()
            real_data_n = real_data_n / real_data_n.max()
            fake_np = fake_sample1.cpu().numpy(); real_np = real_data_n.cpu().numpy()
            val_l1_loss[0, epoch, iteration] = abs(fake_np - real_np).mean()
            val_psnr_values[0, epoch, iteration] = psnr(real_np, fake_np, data_range=1.0)
            val_ssim_values[0, epoch, iteration] = compute_ssim_slices(fake_np, real_np, data_range=1.0)
            val_raw_psnr[0, epoch, iteration] = compute_nrmse(fake_raw, real_raw)
            val_raw_ssim[0, epoch, iteration] = compute_ssim_official(fake_raw, real_raw)

        for iteration, (y_val, x_val) in enumerate(data_loader_val):
            real_data = x_val.to(device, non_blocking=True)
            source_data = y_val.to(device, non_blocking=True)
            x1_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
            fake_sample1 = sample_from_model(pos_coeff, gen1, args.num_timesteps, x1_t, T, args)
            fake_sample1 = to_range_0_1(fake_sample1)
            real_data_n = to_range_0_1(real_data)
            fake_raw = fake_sample1.cpu().numpy(); real_raw = real_data_n.cpu().numpy()
            fake_sample1 = fake_sample1 / fake_sample1.max()
            real_data_n = real_data_n / real_data_n.max()
            fake_np = fake_sample1.cpu().numpy(); real_np = real_data_n.cpu().numpy()
            val_l1_loss[1, epoch, iteration] = abs(fake_np - real_np).mean()
            val_psnr_values[1, epoch, iteration] = psnr(real_np, fake_np, data_range=1.0)
            val_ssim_values[1, epoch, iteration] = compute_ssim_slices(fake_np, real_np, data_range=1.0)
            val_raw_psnr[1, epoch, iteration] = compute_nrmse(fake_raw, real_raw)
            val_raw_ssim[1, epoch, iteration] = compute_ssim_official(fake_raw, real_raw)

        if rank == 0:
            p0 = np.nanmean(val_psnr_values[0, epoch, :])
            p1 = np.nanmean(val_psnr_values[1, epoch, :])
            s0 = np.nanmean(val_ssim_values[0, epoch, :])
            s1 = np.nanmean(val_ssim_values[1, epoch, :])
            rp0 = np.nanmean(val_raw_psnr[0, epoch, :])
            rp1 = np.nanmean(val_raw_psnr[1, epoch, :])
            rs0 = np.nanmean(val_raw_ssim[0, epoch, :])
            rs1 = np.nanmean(val_raw_ssim[1, epoch, :])
            print(f'max-norm  | PSNR dir1={p0:.2f} dir2={p1:.2f}')
            print(f'official  | nRMSE dir1={rp0:.4f} dir2={rp1:.4f} | SSIM dir1={rs0:.4f} dir2={rs1:.4f}')
            np.save(os.path.join(exp_path, 'val_l1_loss.npy'), val_l1_loss)
            np.save(os.path.join(exp_path, 'val_psnr_values.npy'), val_psnr_values)
            np.save(os.path.join(exp_path, 'val_ssim_values.npy'), val_ssim_values)
            metrics_csv = os.path.join(exp_path, 'metrics.csv')
            write_header = not os.path.exists(metrics_csv)
            with open(metrics_csv, 'a') as f:
                if write_header:
                    f.write('epoch,psnr_norm_d1,psnr_norm_d2,ssim_norm_d1,ssim_norm_d2,nrmse_d1,nrmse_d2,ssim_official_d1,ssim_official_d2\n')
                f.write(f'{epoch},{p0:.4f},{p1:.4f},{s0:.4f},{s1:.4f},{rp0:.4f},{rp1:.4f},{rs0:.4f},{rs1:.4f}\n')

        dist.barrier()
        rank_log(args, rank, f'epoch {epoch} complete')

    rank_log(args, rank, 'final barrier enter')
    dist.barrier()
    rank_log(args, rank, 'final barrier exit')
    dist.destroy_process_group()


def init_processes(rank, size, fn, args, local_rank):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(local_rank)
    gpu = local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Experiment A: SynDiff + baseline npz data')
    # Experiment A specific args
    parser.add_argument('--preprocessed_dir', required=True, help='Path to npz cache root')
    parser.add_argument('--src_field', default='1.5T')
    parser.add_argument('--tgt_field', default='7T')
    # SynDiff model args (same as train.py)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=2)
    parser.add_argument('--centered', action='store_false', default=True)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
    parser.add_argument('--num_channels_dae', type=int, default=64)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,1,2,2,4,4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attn_resolutions', default=(16,))
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--resamp_with_conv', action='store_false', default=True)
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--fir', action='store_false', default=True)
    parser.add_argument('--fir_kernel', default=[1,3,3,1])
    parser.add_argument('--skip_rescale', action='store_false', default=True)
    parser.add_argument('--resblock_type', default='biggan')
    parser.add_argument('--progressive', default='none')
    parser.add_argument('--progressive_input', default='residual')
    parser.add_argument('--progressive_combine', default='sum')
    parser.add_argument('--embedding_type', default='positional')
    parser.add_argument('--fourier_scale', type=float, default=16.)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
    parser.add_argument('--exp', default='exp_A_npz_V100x8_15T_7T')
    parser.add_argument('--output_path', default='/NAS_writeable/SynDiff/checkpoints')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=1.6e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--no_lr_decay', action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--r1_gamma', type=float, default=1.0)
    parser.add_argument('--lazy_reg', type=int, default=10)
    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=1)
    parser.add_argument('--save_ckpt_every', type=int, default=1)
    parser.add_argument('--lambda_l1_loss', type=float, default=0.5)
    parser.add_argument('--num_proc_node', type=int, default=1)
    parser.add_argument('--num_process_per_node', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--master_address', type=str, default='127.0.0.1')
    parser.add_argument('--port_num', type=str, default='6021')
    parser.add_argument('--debug_sync_trace', action='store_true', default=False)

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            process_args = copy.deepcopy(args)
            global_rank = rank + args.node_rank * args.num_process_per_node
            p = Process(target=init_processes, args=(global_rank, args.world_size, train_syndiff, process_args, rank))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        init_processes(0, size, train_syndiff, args, 0)
