

import argparse
import copy
import datetime
import torch
import numpy as np

import os

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from dataset import CreateDatasetSynthesis

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_nrmse(pred, target):
    """Official MRIxFields metric: ||pred-target|| / ||target||."""
    diff = pred.astype(np.float64) - target.astype(np.float64)
    norm_t = np.linalg.norm(target.astype(np.float64))
    return float(np.linalg.norm(diff) / norm_t) if norm_t > 1e-10 else 0.0


def _as_2d_slices(array):
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError('SSIM expects at least 2 image dimensions, got {}'.format(array.shape))
    if array.ndim == 2:
        return array.reshape((1,) + array.shape)
    return array.reshape((-1,) + array.shape[-2:])


def compute_ssim_slices(pred, target, data_range=1.0):
    """Mean SSIM over 2D slices; leading dimensions are batch/channel axes."""
    pred_slices = _as_2d_slices(pred)
    target_slices = _as_2d_slices(target)
    if pred_slices.shape != target_slices.shape:
        raise ValueError('SSIM shape mismatch: {} vs {}'.format(pred_slices.shape, target_slices.shape))
    values = [ssim(t, p, data_range=data_range) for p, t in zip(pred_slices, target_slices)]
    return float(np.mean(values)) if values else np.nan


def compute_ssim_official(pred, target):
    """Official MRIxFields SSIM per 2D slice: data_range = target.max() - target.min()."""
    pred_slices = _as_2d_slices(pred)
    target_slices = _as_2d_slices(target)
    if pred_slices.shape != target_slices.shape:
        raise ValueError('SSIM shape mismatch: {} vs {}'.format(pred_slices.shape, target_slices.shape))
    values = []
    for p, t in zip(pred_slices, target_slices):
        dr = t.max() - t.min()
        values.append(1.0 if dr < 1e-10 else ssim(t, p, data_range=dr))
    return float(np.mean(values)) if values else np.nan



def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
            
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def rank_log(args, rank, message):
    if getattr(args, 'debug_sync_trace', False):
        print('[rank{}] {}'.format(rank, message), flush=True)


#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

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
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one
#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init[:,[0],:]
    source = x_init[:,[1],:]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
            x_0 = generator(torch.cat((x,source),axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:,[0],:], x, t)
            x = x_new.detach()
        
    return x

#%%
def train_syndiff(rank, gpu, args):

    
    from backbones.discriminator import Discriminator_small, Discriminator_large
    
    from backbones.ncsnpp_generator_adagn import NCSNpp
    
    import backbones.generator_resnet 
    
    
    from utils.EMA import EMA
    
    #rank = args.node_rank * args.num_process_per_node + gpu
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    
    nz = args.nz #latent dimension
    

    dataset = CreateDatasetSynthesis(phase = "train", input_path = args.input_path, contrast1 = args.contrast1, contrast2 = args.contrast2)
    dataset_val = CreateDatasetSynthesis(phase = "val", input_path = args.input_path, contrast1 = args.contrast1, contrast2 = args.contrast2 )


    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True,
                                               persistent_workers=args.num_workers > 0)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=val_sampler,
                                               drop_last = True,
                                               persistent_workers=args.num_workers > 0)

    val_l1_loss=np.zeros([2,args.num_epoch+1,len(data_loader_val)])
    val_psnr_values=np.zeros([2,args.num_epoch+1,len(data_loader_val)])
    val_ssim_values=np.zeros([2,args.num_epoch+1,len(data_loader_val)])
    val_raw_psnr=np.zeros([2,args.num_epoch+1,len(data_loader_val)])
    val_raw_ssim=np.zeros([2,args.num_epoch+1,len(data_loader_val)])
    print('train data size:'+str(len(data_loader)))
    print('val data size:'+str(len(data_loader_val)))
    to_range_0_1 = lambda x: (x + 1.) / 2.

    #networks performing reverse denoising
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)  
    #networks performing translation
    args.num_channels=1
    gen_non_diffusive_1to2 = backbones.generator_resnet.define_G(netG='resnet_6blocks',gpu_ids=[gpu])
    gen_non_diffusive_2to1 = backbones.generator_resnet.define_G(netG='resnet_6blocks',gpu_ids=[gpu])
    
    disc_diffusive_1 = Discriminator_large(nc = 2, ngf = args.ngf, 
                                   t_emb_dim = args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    disc_diffusive_2 = Discriminator_large(nc = 2, ngf = args.ngf, 
                                   t_emb_dim = args.t_emb_dim,
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
    
    optimizer_disc_diffusive_1 = optim.Adam(disc_diffusive_1.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizer_disc_diffusive_2 = optim.Adam(disc_diffusive_2.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    
    optimizer_gen_diffusive_1 = optim.Adam(gen_diffusive_1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_gen_diffusive_2 = optim.Adam(gen_diffusive_2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    
    optimizer_gen_non_diffusive_1to2 = optim.Adam(gen_non_diffusive_1to2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_gen_non_diffusive_2to1 = optim.Adam(gen_non_diffusive_2to1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))

    optimizer_disc_non_diffusive_cycle1 = optim.Adam(disc_non_diffusive_cycle1.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizer_disc_non_diffusive_cycle2 = optim.Adam(disc_non_diffusive_cycle2.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))    
    
    if args.use_ema:
        optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=args.ema_decay)
        optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=args.ema_decay)
        optimizer_gen_non_diffusive_1to2 = EMA(optimizer_gen_non_diffusive_1to2, ema_decay=args.ema_decay)
        optimizer_gen_non_diffusive_2to1 = EMA(optimizer_gen_non_diffusive_2to1, ema_decay=args.ema_decay)
        
    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_1to2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_1to2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_2to1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_2to1, args.num_epoch, eta_min=1e-5)    
    
    scheduler_disc_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_2, args.num_epoch, eta_min=1e-5)

    scheduler_disc_non_diffusive_cycle1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_non_diffusive_cycle2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle2, args.num_epoch, eta_min=1e-5)

    # AMP: mixed precision via autocast + GradScaler for generator losses
    scaler_g = torch.cuda.amp.GradScaler()


    
    #ddp
    gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, device_ids=[gpu], find_unused_parameters=True)
    gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, device_ids=[gpu], find_unused_parameters=True)
    gen_non_diffusive_1to2 = nn.parallel.DistributedDataParallel(gen_non_diffusive_1to2, device_ids=[gpu], find_unused_parameters=True)
    gen_non_diffusive_2to1 = nn.parallel.DistributedDataParallel(gen_non_diffusive_2to1, device_ids=[gpu], find_unused_parameters=True)
    disc_diffusive_1 = nn.parallel.DistributedDataParallel(disc_diffusive_1, device_ids=[gpu], find_unused_parameters=True)
    disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, device_ids=[gpu], find_unused_parameters=True)

    disc_non_diffusive_cycle1 = nn.parallel.DistributedDataParallel(disc_non_diffusive_cycle1, device_ids=[gpu], find_unused_parameters=True)
    disc_non_diffusive_cycle2 = nn.parallel.DistributedDataParallel(disc_non_diffusive_cycle2, device_ids=[gpu], find_unused_parameters=True)
    
    exp = args.exp
    output_path = args.output_path

    exp_path = os.path.join(output_path,exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('./backbones', os.path.join(exp_path, 'backbones'))
    
    
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
        gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])
        gen_non_diffusive_1to2.load_state_dict(checkpoint['gen_non_diffusive_1to2_dict'])
        gen_non_diffusive_2to1.load_state_dict(checkpoint['gen_non_diffusive_2to1_dict'])        
        # load G
        
        optimizer_gen_diffusive_1.load_state_dict(checkpoint['optimizer_gen_diffusive_1'])
        scheduler_gen_diffusive_1.load_state_dict(checkpoint['scheduler_gen_diffusive_1'])
        optimizer_gen_diffusive_2.load_state_dict(checkpoint['optimizer_gen_diffusive_2'])
        scheduler_gen_diffusive_2.load_state_dict(checkpoint['scheduler_gen_diffusive_2']) 
        optimizer_gen_non_diffusive_1to2.load_state_dict(checkpoint['optimizer_gen_non_diffusive_1to2'])
        scheduler_gen_non_diffusive_1to2.load_state_dict(checkpoint['scheduler_gen_non_diffusive_1to2'])
        optimizer_gen_non_diffusive_2to1.load_state_dict(checkpoint['optimizer_gen_non_diffusive_2to1'])
        scheduler_gen_non_diffusive_2to1.load_state_dict(checkpoint['scheduler_gen_non_diffusive_2to1'])          
        # load D
        disc_diffusive_1.load_state_dict(checkpoint['disc_diffusive_1_dict'])
        optimizer_disc_diffusive_1.load_state_dict(checkpoint['optimizer_disc_diffusive_1'])
        scheduler_disc_diffusive_1.load_state_dict(checkpoint['scheduler_disc_diffusive_1'])

        disc_diffusive_2.load_state_dict(checkpoint['disc_diffusive_2_dict'])
        optimizer_disc_diffusive_2.load_state_dict(checkpoint['optimizer_disc_diffusive_2'])
        scheduler_disc_diffusive_2.load_state_dict(checkpoint['scheduler_disc_diffusive_2'])   
        # load D_for cycle
        disc_non_diffusive_cycle1.load_state_dict(checkpoint['disc_non_diffusive_cycle1_dict'])
        optimizer_disc_non_diffusive_cycle1.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle1'])
        scheduler_disc_non_diffusive_cycle1.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle1'])

        disc_non_diffusive_cycle2.load_state_dict(checkpoint['disc_non_diffusive_cycle2_dict'])
        optimizer_disc_non_diffusive_cycle2.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle2'])
        scheduler_disc_non_diffusive_cycle2.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle2'])
        global_step = checkpoint['global_step']
        if 'scaler_g' in checkpoint:
            scaler_g.load_state_dict(checkpoint['scaler_g'])
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
    
    for epoch in range(init_epoch + 1, args.num_epoch+1):
        rank_log(args, rank, 'epoch {} train start'.format(epoch))
        train_sampler.set_epoch(epoch)
       
        for iteration, (x1, x2) in enumerate(data_loader):
            for p in disc_diffusive_1.parameters():  
                p.requires_grad = True  
            for p in disc_diffusive_2.parameters():  
                p.requires_grad = True
            for p in disc_non_diffusive_cycle1.parameters():  
                p.requires_grad = True  
            for p in disc_non_diffusive_cycle2.parameters():  
                p.requires_grad = True          
            
            disc_diffusive_1.zero_grad()
            disc_diffusive_2.zero_grad()
            
            #sample from p(x_0)
            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)
            
            #sample t
            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            #sample x_t and x_tp1
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x1_t.requires_grad = True
            
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
            x2_t.requires_grad = True               
            # train discriminator with real
            D1_real = disc_diffusive_1(x1_t, t1, x1_tp1.detach()).view(-1)
            D2_real = disc_diffusive_2(x2_t, t2, x2_tp1.detach()).view(-1)

            errD1_real = F.softplus(-D1_real.float())
            errD1_real = errD1_real.mean()

            errD2_real = F.softplus(-D2_real.float())
            errD2_real = errD2_real.mean()
            errD_real = errD1_real + errD2_real
            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad1_real = torch.autograd.grad(
                            outputs=D1_real.float().sum(), inputs=x1_t, create_graph=True
                            )[0]
                grad1_penalty = (
                                grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                grad2_real = torch.autograd.grad(
                            outputs=D2_real.float().sum(), inputs=x2_t, create_graph=True
                            )[0]
                grad2_penalty = (
                                grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()

                grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad1_real = torch.autograd.grad(
                            outputs=D1_real.float().sum(), inputs=x1_t, create_graph=True
                            )[0]
                    grad1_penalty = (
                                grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                    grad2_real = torch.autograd.grad(
                            outputs=D2_real.float().sum(), inputs=x2_t, create_graph=True
                            )[0]
                    grad2_penalty = (
                                grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
                    grad_penalty.backward()


            # train with fake
            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)

            # no_sync: skip gradient all-reduce for generators during D phase —
            # these grads are zeroed before G phase, so syncing them is wasted comm.
            with gen_diffusive_1.no_sync(), gen_diffusive_2.no_sync(), \
                 gen_non_diffusive_1to2.no_sync(), gen_non_diffusive_2to1.no_sync():
                # Generators in fp32 (AdaGN underflows in fp16); discriminators below in fp16
                x1_0_predict = gen_non_diffusive_2to1(real_data2)
                x2_0_predict = gen_non_diffusive_1to2(real_data1)
                #x_tp1 is concatenated with source contrast and x_0_predict is predicted
                x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(),x2_0_predict),axis=1), t1, latent_z1)
                x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(),x1_0_predict),axis=1), t2, latent_z2)
                #sampling q(x_t | x_0_predict, x_t+1)
                x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:,[0],:], x1_tp1, t1)
                x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:,[0],:], x2_tp1, t2)
                #D output for fake sample x_pos_sample (discriminators in fp16)
                output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
                output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)

                errD1_fake = F.softplus(output1.float())
                errD2_fake = F.softplus(output2.float())
                errD_fake = errD1_fake.mean() + errD2_fake.mean()
                errD_fake.backward()    
            
            errD = errD_real + errD_fake
            # Update D
            torch.nn.utils.clip_grad_norm_(disc_diffusive_1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(disc_diffusive_2.parameters(), max_norm=1.0)
            optimizer_disc_diffusive_1.step()
            optimizer_disc_diffusive_2.step()  

            #D for cycle part
            disc_non_diffusive_cycle1.zero_grad()
            disc_non_diffusive_cycle2.zero_grad()

            D_cycle1_real = disc_non_diffusive_cycle1(real_data1).view(-1)
            D_cycle2_real = disc_non_diffusive_cycle2(real_data2).view(-1)

            errD_cycle1_real = F.softplus(-D_cycle1_real.float())
            errD_cycle1_real = errD_cycle1_real.mean()

            errD_cycle2_real = F.softplus(-D_cycle2_real.float())
            errD_cycle2_real = errD_cycle2_real.mean()
            errD_cycle_real = errD_cycle1_real + errD_cycle2_real
            errD_cycle_real.backward(retain_graph=True)
            # train with fake
            x1_0_predict_cycle = gen_non_diffusive_2to1(real_data2)
            x2_0_predict_cycle = gen_non_diffusive_1to2(real_data1)
            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict_cycle).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict_cycle).view(-1)

            errD_cycle1_fake = F.softplus(D_cycle1_fake.float())
            errD_cycle1_fake = errD_cycle1_fake.mean()

            errD_cycle2_fake = F.softplus(D_cycle2_fake.float())
            errD_cycle2_fake = errD_cycle2_fake.mean()
            errD_cycle_fake = errD_cycle1_fake + errD_cycle2_fake
            errD_cycle_fake.backward()

            errD_cycle = errD_cycle_real + errD_cycle_fake
            # Update D
            torch.nn.utils.clip_grad_norm_(disc_non_diffusive_cycle1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(disc_non_diffusive_cycle2.parameters(), max_norm=1.0)
            optimizer_disc_non_diffusive_cycle1.step()
            optimizer_disc_non_diffusive_cycle2.step() 

            #G part
            for p in disc_diffusive_1.parameters():
                p.requires_grad = False
            for p in disc_diffusive_2.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle1.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle2.parameters():
                p.requires_grad = False                
            gen_diffusive_1.zero_grad()
            gen_diffusive_2.zero_grad()
            gen_non_diffusive_1to2.zero_grad()
            gen_non_diffusive_2to1.zero_grad()   
            
            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            
            #sample x_t and x_tp1            
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)   
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)             
            
            latent_z1 = torch.randn(batch_size, nz,device=device)
            latent_z2 = torch.randn(batch_size, nz,device=device)
            
            #translation networks (generators in fp32 — AdaGN underflows in fp16)
            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict_cycle = gen_non_diffusive_1to2(x1_0_predict)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)
            x1_0_predict_cycle = gen_non_diffusive_2to1(x2_0_predict)

            #x_tp1 is concatenated with source contrast and x_0_predict is predicted
            x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(),x2_0_predict),axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(),x1_0_predict),axis=1), t2, latent_z2)
            #sampling q(x_t | x_0_predict, x_t+1)
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:,[0],:], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:,[0],:], x2_tp1, t2)
            #D output for fake sample x_pos_sample
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)

            #D_cycle output for fake x1_0_predict
            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1)


            errG1 = F.softplus(-output1.float())
            errG1 = errG1.mean()

            errG2 = F.softplus(-output2.float())
            errG2 = errG2.mean()

            errG_adv = errG1 + errG2

            errG_cycle_adv1 = F.softplus(-D_cycle1_fake.float())
            errG_cycle_adv1 = errG_cycle_adv1.mean()

            errG_cycle_adv2 = F.softplus(-D_cycle2_fake.float())
            errG_cycle_adv2 = errG_cycle_adv2.mean()
            errG_cycle_adv = errG_cycle_adv1 + errG_cycle_adv2

            #L1 loss
            errG1_L1 = F.l1_loss(x1_0_predict_diff[:,[0],:],real_data1)
            errG2_L1 = F.l1_loss(x2_0_predict_diff[:,[0],:],real_data2)
            errG_L1 = errG1_L1 + errG2_L1

            #cycle loss
            errG1_cycle=F.l1_loss(x1_0_predict_cycle,real_data1)
            errG2_cycle=F.l1_loss(x2_0_predict_cycle,real_data2)
            errG_cycle = errG1_cycle + errG2_cycle

            errG = args.lambda_l1_loss*errG_cycle +  errG_adv + errG_cycle_adv + args.lambda_l1_loss*errG_L1
            scaler_g.scale(errG).backward()

            scaler_g.unscale_(optimizer_gen_diffusive_1)
            scaler_g.unscale_(optimizer_gen_diffusive_2)
            scaler_g.unscale_(optimizer_gen_non_diffusive_1to2)
            scaler_g.unscale_(optimizer_gen_non_diffusive_2to1)

            torch.nn.utils.clip_grad_norm_(gen_diffusive_1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(gen_diffusive_2.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(gen_non_diffusive_1to2.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(gen_non_diffusive_2to1.parameters(), max_norm=1.0)

            optimizer_gen_diffusive_1.step()
            optimizer_gen_diffusive_2.step()
            optimizer_gen_non_diffusive_1to2.step()
            optimizer_gen_non_diffusive_2to1.step()

            scaler_g.update()           
            
            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, G-Cycle: {}, G-L1: {}, G-Adv: {}, G-cycle-Adv: {}, G-Sum: {}, D Loss: {}, D_cycle Loss: {}'.format(epoch,iteration, errG_cycle.item(), errG_L1.item(),  errG_adv.item(), errG_cycle_adv.item(), errG.item(), errD.item(), errD_cycle.item()))
        
        if not args.no_lr_decay:
            
            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()
            scheduler_gen_non_diffusive_1to2.step()
            scheduler_gen_non_diffusive_2to1.step()
            scheduler_disc_diffusive_1.step()
            scheduler_disc_diffusive_2.step()

            scheduler_disc_non_diffusive_cycle1.step()
            scheduler_disc_non_diffusive_cycle2.step()
        
        rank_log(args, rank, 'epoch {} train complete'.format(epoch))

        # Use underlying (non-DDP) models for sample generation to avoid
        # NCCL parameter broadcasts that would hang other ranks.
        gen1 = gen_diffusive_1.module if hasattr(gen_diffusive_1, 'module') else gen_diffusive_1
        gen2 = gen_diffusive_2.module if hasattr(gen_diffusive_2, 'module') else gen_diffusive_2
        nd1to2 = gen_non_diffusive_1to2.module if hasattr(gen_non_diffusive_1to2, 'module') else gen_non_diffusive_1to2
        nd2to1 = gen_non_diffusive_2to1.module if hasattr(gen_non_diffusive_2to1, 'module') else gen_non_diffusive_2to1

        if rank == 0:
            rank_log(args, rank, 'epoch {} rank0 sample/save start'.format(epoch))
            grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
            if epoch % 10 == 0:
                torchvision.utils.save_image(x1_pos_sample, os.path.join(exp_path, 'xpos1_epoch_{}.png'.format(epoch)), normalize=True)
                torchvision.utils.save_image(x2_pos_sample, os.path.join(exp_path, 'xpos2_epoch_{}.png'.format(epoch)), normalize=True)
            #concatenate noise and source contrast
            x1_t = torch.cat((torch.randn_like(real_data1),real_data2),axis=1)
            fake_sample1 = sample_from_model(pos_coeff, gen1, args.num_timesteps, x1_t, T, args)
            fake_sample1 = torch.cat((real_data2, fake_sample1),axis=-1)
            torchvision.utils.save_image(fake_sample1, os.path.join(exp_path, 'sample1_discrete_epoch_{}.png'.format(epoch)), normalize=True)
            pred1 = nd2to1(real_data2)
            #
            x2_t = torch.cat((torch.randn_like(real_data2), pred1),axis=1)
            fake_sample2_tilda = gen2(x2_t , t2, latent_z2)
            #
            pred1 = torch.cat((real_data2, pred1, nd1to2(pred1), fake_sample2_tilda[:,[0],:]),axis=-1)
            torchvision.utils.save_image(pred1, os.path.join(exp_path, 'sample1_translated_epoch_{}.png'.format(epoch)), normalize=True)


            x2_t = torch.cat((torch.randn_like(real_data2),real_data1),axis=1)
            fake_sample2 = sample_from_model(pos_coeff, gen2, args.num_timesteps, x2_t, T, args)
            fake_sample2 = torch.cat((real_data1, fake_sample2),axis=-1)
            torchvision.utils.save_image(fake_sample2, os.path.join(exp_path, 'sample2_discrete_epoch_{}.png'.format(epoch)), normalize=True)
            pred2 = nd1to2(real_data1)
            #
            x1_t = torch.cat((torch.randn_like(real_data1), pred2),axis=1)
            fake_sample1_tilda = gen1(x1_t , t1, latent_z1)
            #
            pred2 = torch.cat((real_data1, pred2, nd2to1(pred2), fake_sample1_tilda[:,[0],:]),axis=-1)
            torchvision.utils.save_image(pred2, os.path.join(exp_path, 'sample2_translated_epoch_{}.png'.format(epoch)), normalize=True)
            torch.set_grad_enabled(grad_enabled)
           
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'gen_diffusive_1_dict': gen_diffusive_1.state_dict(), 'optimizer_gen_diffusive_1': optimizer_gen_diffusive_1.state_dict(),
                               'gen_diffusive_2_dict': gen_diffusive_2.state_dict(), 'optimizer_gen_diffusive_2': optimizer_gen_diffusive_2.state_dict(),
                               'scheduler_gen_diffusive_1': scheduler_gen_diffusive_1.state_dict(), 'disc_diffusive_1_dict': disc_diffusive_1.state_dict(),
                               'scheduler_gen_diffusive_2': scheduler_gen_diffusive_2.state_dict(), 'disc_diffusive_2_dict': disc_diffusive_2.state_dict(),
                               'gen_non_diffusive_1to2_dict': gen_non_diffusive_1to2.state_dict(), 'optimizer_gen_non_diffusive_1to2': optimizer_gen_non_diffusive_1to2.state_dict(),
                               'gen_non_diffusive_2to1_dict': gen_non_diffusive_2to1.state_dict(), 'optimizer_gen_non_diffusive_2to1': optimizer_gen_non_diffusive_2to1.state_dict(),
                               'scheduler_gen_non_diffusive_1to2': scheduler_gen_non_diffusive_1to2.state_dict(), 'scheduler_gen_non_diffusive_2to1': scheduler_gen_non_diffusive_2to1.state_dict(),
                               'optimizer_disc_diffusive_1': optimizer_disc_diffusive_1.state_dict(), 'scheduler_disc_diffusive_1': scheduler_disc_diffusive_1.state_dict(),
                               'optimizer_disc_diffusive_2': optimizer_disc_diffusive_2.state_dict(), 'scheduler_disc_diffusive_2': scheduler_disc_diffusive_2.state_dict(),
                               'optimizer_disc_non_diffusive_cycle1': optimizer_disc_non_diffusive_cycle1.state_dict(), 'scheduler_disc_non_diffusive_cycle1': scheduler_disc_non_diffusive_cycle1.state_dict(),
                               'optimizer_disc_non_diffusive_cycle2': optimizer_disc_non_diffusive_cycle2.state_dict(), 'scheduler_disc_non_diffusive_cycle2': scheduler_disc_non_diffusive_cycle2.state_dict(),
                               'disc_non_diffusive_cycle1_dict': disc_non_diffusive_cycle1.state_dict(),'disc_non_diffusive_cycle2_dict': disc_non_diffusive_cycle2.state_dict(),
                               'scaler_g': scaler_g.state_dict()}

                    torch.save(content, os.path.join(exp_path, 'content.pth'))
                    del content
                    torch.cuda.empty_cache()
                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)                    
                torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1_{}.pth'.format(epoch)))
                torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2_{}.pth'.format(epoch)))
                torch.save(gen_non_diffusive_1to2.state_dict(), os.path.join(exp_path, 'gen_non_diffusive_1to2_{}.pth'.format(epoch)))
                torch.save(gen_non_diffusive_2to1.state_dict(), os.path.join(exp_path, 'gen_non_diffusive_2to1_{}.pth'.format(epoch)))
                torch.cuda.empty_cache()                
                if args.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)
            rank_log(args, rank, 'epoch {} rank0 sample/save complete'.format(epoch))


        rank_log(args, rank, 'epoch {} pre-validation barrier enter'.format(epoch))
        dist.barrier()
        rank_log(args, rank, 'epoch {} validation start'.format(epoch))
        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        for iteration, (x_val , y_val) in enumerate(data_loader_val): 
        
            real_data = x_val.to(device, non_blocking=True)
            source_data = y_val.to(device, non_blocking=True)
            
            x1_t = torch.cat((torch.randn_like(real_data),source_data),axis=1)
            #diffusion steps
            fake_sample1 = sample_from_model(pos_coeff, gen1, args.num_timesteps, x1_t, T, args)
            fake_sample1 = to_range_0_1(fake_sample1)
            real_data = to_range_0_1(real_data)
            # raw metrics (no max-normalization, preserves intensity differences)
            fake_raw = fake_sample1.cpu().numpy()
            real_raw = real_data.cpu().numpy()
            # max-normalized metrics (structure-focused, matches test.py)
            fake_sample1 = fake_sample1 / fake_sample1.max()
            real_data = real_data / real_data.max()
            fake_sample1=fake_sample1.cpu().numpy()
            real_data=real_data.cpu().numpy()
            # Per-slice metrics: max-norm (matching test.py) + official (raw)
            val_l1_loss[0,epoch,iteration]=abs(fake_sample1 - real_data).mean()
            val_psnr_values[0,epoch, iteration] = psnr(real_data, fake_sample1, data_range=1.0)
            val_ssim_values[0,epoch, iteration] = compute_ssim_slices(fake_sample1, real_data, data_range=1.0)
            # Official metrics on raw [0,1] data (no max-normalization)
            val_raw_psnr[0,epoch, iteration]  = compute_nrmse(fake_raw, real_raw)
            val_raw_ssim[0,epoch, iteration]  = compute_ssim_official(fake_raw, real_raw)

        for iteration, (y_val , x_val) in enumerate(data_loader_val):

            real_data = x_val.to(device, non_blocking=True)
            source_data = y_val.to(device, non_blocking=True)

            x1_t = torch.cat((torch.randn_like(real_data),source_data),axis=1)
            #diffusion steps
            fake_sample1 = sample_from_model(pos_coeff, gen1, args.num_timesteps, x1_t, T, args)


            fake_sample1 = to_range_0_1(fake_sample1)
            real_data = to_range_0_1(real_data)
            # raw metrics (no max-normalization)
            fake_raw = fake_sample1.cpu().numpy()
            real_raw = real_data.cpu().numpy()
            # max-normalized metrics (structure-focused)
            fake_sample1 = fake_sample1 / fake_sample1.max()
            real_data = real_data / real_data.max()
            fake_sample1=fake_sample1.cpu().numpy()
            real_data=real_data.cpu().numpy()
            # Per-slice metrics: max-norm (matching test.py) + official (raw)
            val_l1_loss[1,epoch,iteration]=abs(fake_sample1 - real_data).mean()
            val_psnr_values[1,epoch, iteration] = psnr(real_data, fake_sample1, data_range=1.0)
            val_ssim_values[1,epoch, iteration] = compute_ssim_slices(fake_sample1, real_data, data_range=1.0)
            # Official metrics on raw [0,1] data (no max-normalization)
            val_raw_psnr[1,epoch, iteration]  = compute_nrmse(fake_raw, real_raw)
            val_raw_ssim[1,epoch, iteration]  = compute_ssim_official(fake_raw, real_raw)
        torch.set_grad_enabled(grad_enabled)
        rank_log(args, rank, 'epoch {} validation complete'.format(epoch))

        if rank == 0:
            p0 = np.nanmean(val_psnr_values[0,epoch,:])
            p1 = np.nanmean(val_psnr_values[1,epoch,:])
            s0 = np.nanmean(val_ssim_values[0,epoch,:])
            s1 = np.nanmean(val_ssim_values[1,epoch,:])
            mae0 = np.nanmean(val_l1_loss[0,epoch,:])
            mae1 = np.nanmean(val_l1_loss[1,epoch,:])
            rp0 = np.nanmean(val_raw_psnr[0,epoch,:])
            rp1 = np.nanmean(val_raw_psnr[1,epoch,:])
            rs0 = np.nanmean(val_raw_ssim[0,epoch,:])
            rs1 = np.nanmean(val_raw_ssim[1,epoch,:])
            print('max-norm  | PSNR dir1={:.2f} dir2={:.2f} | SSIM dir1={:.4f} dir2={:.4f} | MAE dir1={:.4f} dir2={:.4f}'.format(p0, p1, s0, s1, mae0, mae1))
            print('official  | nRMSE dir1={:.4f} dir2={:.4f} | SSIM dir1={:.4f} dir2={:.4f}'.format(rp0, rp1, rs0, rs1))
            np.save('{}/val_l1_loss.npy'.format(exp_path), val_l1_loss)
            np.save('{}/val_psnr_values.npy'.format(exp_path), val_psnr_values)
            np.save('{}/val_ssim_values.npy'.format(exp_path), val_ssim_values)
            np.save('{}/val_nrmse.npy'.format(exp_path), val_raw_psnr)
            np.save('{}/val_ssim_official.npy'.format(exp_path), val_raw_ssim)
            # Append per-epoch metrics CSV
            metrics_csv = '{}/metrics.csv'.format(exp_path)
            write_header = not os.path.exists(metrics_csv)
            with open(metrics_csv, 'a') as f:
                if write_header:
                    f.write('epoch,psnr_norm_d1,psnr_norm_d2,ssim_norm_d1,ssim_norm_d2,mae_norm_d1,mae_norm_d2,nrmse_d1,nrmse_d2,ssim_official_d1,ssim_official_d2\n')
                f.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(epoch, p0, p1, s0, s1, mae0, mae1, rp0, rp1, rs0, rs1))

        # Sync all ranks before next epoch — prevents NCCL timeout when ranks
        # finish validation at different times and enter DDP forward out of sync.
        rank_log(args, rank, 'epoch {} post-validation barrier enter'.format(epoch))
        dist.barrier()
        rank_log(args, rank, 'epoch {} complete'.format(epoch))


def init_processes(rank, size, fn, args, local_rank):
    """ Initialize the distributed environment. """
    args.local_rank = local_rank
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    timeout = datetime.timedelta(seconds=args.pg_timeout_sec)
    try:
        dist.init_process_group(backend=args.dist_backend, init_method='env://', rank=rank, world_size=size, timeout=timeout)
        fn(rank, gpu, args)
        rank_log(args, rank, 'final barrier enter')
        dist.barrier()
        rank_log(args, rank, 'final barrier exit')
    finally:
        cleanup()

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    
    parser.add_argument('--resume', action='store_true',default=False)
    
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='ixi_synth', help='name of experiment')
    parser.add_argument('--input_path', help='path to input data')
    parser.add_argument('--output_path', help='path to output saves')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='data loader workers (0 = main process)')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    
    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save ckpt every x epochs')
    parser.add_argument('--lambda_l1_loss', type=float, default=0.5, help='weightening of l1 loss part of diffusion ans cycle models')
    parser.add_argument('--dist_backend', type=str, default='nccl', choices=['nccl', 'gloo'])
    parser.add_argument('--pg_timeout_sec', type=int, default=600, help='process group timeout in seconds')
    parser.add_argument('--debug_sync_trace', action='store_true', default=False, help='print rank-level epoch-boundary trace messages')
   
    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--contrast1', type=str, default='T1',
                        help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='T2',
                        help='contrast selection for model')
    parser.add_argument('--port_num', type=str, default='6021',
                        help='port selection for code')

   
    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            process_args = copy.deepcopy(args)
            process_args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            process_args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train_syndiff, process_args, rank))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        failed = [(p.pid, p.exitcode) for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError('Distributed worker(s) failed: {}'.format(failed))
    else:

        init_processes(0, size, train_syndiff, args, 0)
