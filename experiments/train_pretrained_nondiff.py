#!/usr/bin/env python3
"""Train full SynDiff with non-diffusive generators pre-trained from CycleGAN.

1. Train non-diffusive generators standalone (CycleGAN-style) for K epochs
2. Load those weights into gen_non_diffusive_1to2/2to1
3. Train full SynDiff (diffusive + non-diffusive) with lambda_l1=10.0 for M epochs

Tests: does properly initialized non-diff fix the SynDiff degradation?
"""
import argparse, os, sys, copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process
import torchvision
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
import backbones.generator_resnet
from dataset import CreateDatasetSynthesis
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---- Helpers (same as train.py) ----
def broadcast_params(params):
    for param in params: dist.broadcast(param.data, src=0)

def var_func_vp(t, beta_min, beta_max):
    return 1. - torch.exp(2. * (-0.25*t**2*(beta_max-beta_min) - 0.5*t*beta_min))

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    return out.reshape(shape[0], *[1]*(len(shape)-1))

def get_time_schedule(args, device):
    t = np.arange(0, args.num_timesteps+1, dtype=np.float64)/args.num_timesteps
    return torch.from_numpy(t*(1.-1e-3)+1e-3).to(device)

def get_sigma_schedule(args, device):
    t = np.arange(0, args.num_timesteps+1, dtype=np.float64)/args.num_timesteps
    t = torch.from_numpy(t*(1.-1e-3)+1e-3)
    var = var_func_vp(t, args.beta_min, args.beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:]/alpha_bars[:-1]
    betas = torch.cat((torch.tensor(1e-8)[None], betas)).to(device).float()
    return betas**0.5, torch.sqrt(1-betas), betas

class Diffusion_Coefficients:
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum**2)
        self.a_s_prev = self.a_s.clone(); self.a_s_prev[-1]=1
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)

def q_sample_pairs(coeff, x_start, t):
    noise = torch.randn_like(x_start)
    x_t = extract(coeff.a_s_cum,t,x_start.shape)*x_start + extract(coeff.sigmas_cum,t,x_start.shape)*noise
    x_t_plus_one = extract(coeff.a_s,t+1,x_start.shape)*x_t + extract(coeff.sigmas,t+1,x_start.shape)*noise
    return x_t, x_t_plus_one

class Posterior_Coefficients:
    def __init__(self, args, device):
        _,_,self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas[1:]
        self.alphas = 1-self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.],device=device),self.alphas_cumprod[:-1]),0)
        self.posterior_variance = self.betas*(1-self.alphas_cumprod_prev)/(1-self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas*torch.sqrt(self.alphas_cumprod_prev)/(1-self.alphas_cumprod)
        self.posterior_mean_coef2 = (1-self.alphas_cumprod_prev)*torch.sqrt(self.alphas)/(1-self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def sample_posterior(coeff, x_0, x_t, t):
    mean = extract(coeff.posterior_mean_coef1,t,x_t.shape)*x_0 + extract(coeff.posterior_mean_coef2,t,x_t.shape)*x_t
    log_var = extract(coeff.posterior_log_variance_clipped,t,x_t.shape)
    return mean + (1-(t==0).float())[:,None,None,None]*torch.exp(0.5*log_var)*torch.randn_like(x_t)

def sample_from_model(coeff, gen, n_time, x_init, T, opt):
    x = x_init[:,[0],:]; source = x_init[:,[1],:]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),),i,dtype=torch.int64).to(x.device)
            z = torch.randn(x.size(0),opt.nz,device=x.device)
            x_0 = gen(torch.cat((x,source),1),t,z)
            x = sample_posterior(coeff,x_0[:,[0],:],x,t).detach()
    return x

def compute_nrmse(p,t):
    d=p.astype(np.float64)-t.astype(np.float64);n=np.linalg.norm(t.astype(np.float64))
    return float(np.linalg.norm(d)/n) if n>1e-10 else 0.

def _as_2d(a):
    a=np.asarray(a)
    if a.ndim==2: return a.reshape((1,)+a.shape)
    return a.reshape((-1,)+a.shape[-2:])

def compute_ssim_slices(p,t,dr=1.0):
    ps=_as_2d(p);ts=_as_2d(t)
    return float(np.mean([ssim(t,p,data_range=dr) for p,t in zip(ps,ts)]))

def compute_ssim_official(p,t):
    ps=_as_2d(p);ts=_as_2d(t)
    v=[]
    for p,t in zip(ps,ts):
        dr=t.max()-t.min()
        v.append(1.0 if dr<1e-10 else ssim(t,p,data_range=dr))
    return float(np.mean(v))

# ---- Phase 1: Pre-train non-diffusive generators (CycleGAN-style) ----
def pretrain_nondiff(args, device, rank):
    B = args.batch_size
    dataset = CreateDatasetSynthesis('train', args.input_path, args.contrast1, args.contrast2)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False, num_workers=2,
                                          pin_memory=True, sampler=sampler, drop_last=True)
    gen_12 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[args.local_rank])
    gen_21 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[args.local_rank])
    disc_1 = backbones.generator_resnet.define_D(gpu_ids=[args.local_rank])
    disc_2 = backbones.generator_resnet.define_D(gpu_ids=[args.local_rank])
    broadcast_params(gen_12.parameters()); broadcast_params(gen_21.parameters())
    broadcast_params(disc_1.parameters()); broadcast_params(disc_2.parameters())
    gen_12 = nn.parallel.DistributedDataParallel(gen_12, device_ids=[args.local_rank], find_unused_parameters=True)
    gen_21 = nn.parallel.DistributedDataParallel(gen_21, device_ids=[args.local_rank], find_unused_parameters=True)
    disc_1 = nn.parallel.DistributedDataParallel(disc_1, device_ids=[args.local_rank], find_unused_parameters=True)
    disc_2 = nn.parallel.DistributedDataParallel(disc_2, device_ids=[args.local_rank], find_unused_parameters=True)
    optG = optim.Adam(list(gen_12.parameters())+list(gen_21.parameters()), lr=2e-4, betas=(0.5,0.999))
    optD = optim.Adam(list(disc_1.parameters())+list(disc_2.parameters()), lr=2e-4, betas=(0.5,0.999))

    for epoch in range(1, args.pretrain_epochs+1):
        sampler.set_epoch(epoch)
        for _, (x1,x2) in enumerate(loader):
            r1=x1.to(device); r2=x2.to(device)
            disc_1.zero_grad(); disc_2.zero_grad()
            D1r=disc_1(r1).view(-1); D2r=disc_2(r2).view(-1)
            with torch.no_grad(): f1=gen_21(r2); f2=gen_12(r1)
            D1f=disc_1(f1.detach()).view(-1); D2f=disc_2(f2.detach()).view(-1)
            lossD = (nn.functional.softplus(-D1r.float()).mean()+nn.functional.softplus(-D2r.float()).mean()+
                     nn.functional.softplus(D1f.float()).mean()+nn.functional.softplus(D2f.float()).mean())
            lossD.backward(); optD.step()

            gen_12.zero_grad(); gen_21.zero_grad()
            f1=gen_21(r2); f2=gen_12(r1)
            c1=gen_21(f2); c2=gen_12(f1)
            D1f=disc_1(f1).view(-1); D2f=disc_2(f2).view(-1)
            lossG = (nn.functional.softplus(-D1f.float()).mean()+nn.functional.softplus(-D2f.float()).mean()+
                     10.0*(nn.functional.l1_loss(c1,r1)+nn.functional.l1_loss(c2,r2)))
            lossG.backward(); optG.step()
    # Return state dicts
    g12 = gen_12.module if hasattr(gen_12,'module') else gen_12
    g21 = gen_21.module if hasattr(gen_21,'module') else gen_21
    return g12.state_dict(), g21.state_dict()

# ---- Phase 2: Full SynDiff training ----
def train_syndiff_pretrained(rank, gpu, args, nd_12_sd, nd_21_sd):
    from backbones.discriminator import Discriminator_small, Discriminator_large
    from backbones.ncsnpp_generator_adagn import NCSNpp
    from utils.EMA import EMA
    import shutil

    torch.manual_seed(args.seed+rank); torch.cuda.manual_seed(args.seed+rank)
    device = torch.device(f'cuda:{gpu}'); B = args.batch_size; nz = args.nz

    dataset = CreateDatasetSynthesis('train', args.input_path, args.contrast1, args.contrast2)
    dataset_val = CreateDatasetSynthesis('val', args.input_path, args.contrast1, args.contrast2)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False, num_workers=2,
                                          pin_memory=True, sampler=sampler, drop_last=True)
    sampler_v = torch.utils.data.distributed.DistributedSampler(dataset_val, num_replicas=args.world_size, rank=rank)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2,
                                              pin_memory=True, sampler=sampler_v, drop_last=True)
    to_range_0_1 = lambda x: (x+1.)/2.

    gen_diff_1 = NCSNpp(args).to(device); gen_diff_2 = NCSNpp(args).to(device)
    args.num_channels = 1
    gen_nd_12 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[gpu])
    gen_nd_21 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[gpu])
    # Load pre-trained weights
    gen_nd_12.load_state_dict(nd_12_sd); gen_nd_21.load_state_dict(nd_21_sd)

    disc_d1 = Discriminator_large(nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
    disc_d2 = Discriminator_large(nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
    disc_c1 = backbones.generator_resnet.define_D(gpu_ids=[gpu])
    disc_c2 = backbones.generator_resnet.define_D(gpu_ids=[gpu])

    broadcast_params(gen_diff_1.parameters()); broadcast_params(gen_diff_2.parameters())
    broadcast_params(gen_nd_12.parameters()); broadcast_params(gen_nd_21.parameters())
    broadcast_params(disc_d1.parameters()); broadcast_params(disc_d2.parameters())
    broadcast_params(disc_c1.parameters()); broadcast_params(disc_c2.parameters())

    # DDP
    gen_diff_1 = nn.parallel.DistributedDataParallel(gen_diff_1, device_ids=[gpu], find_unused_parameters=True)
    gen_diff_2 = nn.parallel.DistributedDataParallel(gen_diff_2, device_ids=[gpu], find_unused_parameters=True)
    gen_nd_12 = nn.parallel.DistributedDataParallel(gen_nd_12, device_ids=[gpu], find_unused_parameters=True)
    gen_nd_21 = nn.parallel.DistributedDataParallel(gen_nd_21, device_ids=[gpu], find_unused_parameters=True)
    disc_d1 = nn.parallel.DistributedDataParallel(disc_d1, device_ids=[gpu], find_unused_parameters=True)
    disc_d2 = nn.parallel.DistributedDataParallel(disc_d2, device_ids=[gpu], find_unused_parameters=True)
    disc_c1 = nn.parallel.DistributedDataParallel(disc_c1, device_ids=[gpu], find_unused_parameters=True)
    disc_c2 = nn.parallel.DistributedDataParallel(disc_c2, device_ids=[gpu], find_unused_parameters=True)

    # Optimizers
    opt_d_d1 = optim.Adam(disc_d1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    opt_d_d2 = optim.Adam(disc_d2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    opt_g_d1 = optim.Adam(gen_diff_1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_g_d2 = optim.Adam(gen_diff_2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_g_nd12 = optim.Adam(gen_nd_12.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_g_nd21 = optim.Adam(gen_nd_21.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d_c1 = optim.Adam(disc_c1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    opt_d_c2 = optim.Adam(disc_c2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    exp_path = os.path.join(args.output_path, args.exp)
    if rank == 0:
        os.makedirs(exp_path, exist_ok=True)
        shutil.copytree(os.path.join(CODE_DIR,'backbones'), os.path.join(exp_path,'backbones'), dirs_exist_ok=True)

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    for epoch in range(1, args.num_epoch+1):
        sampler.set_epoch(epoch)
        for _, (x1,x2) in enumerate(loader):
            r1=x1.to(device); r2=x2.to(device)

            # --- D diffusive ---
            disc_d1.zero_grad(); disc_d2.zero_grad()
            t1=torch.randint(0,args.num_timesteps,(B,),device=device)
            t2=torch.randint(0,args.num_timesteps,(B,),device=device)
            x1_t,x1_tp1=q_sample_pairs(coeff,r1,t1); x1_t.requires_grad=True
            x2_t,x2_tp1=q_sample_pairs(coeff,r2,t2); x2_t.requires_grad=True
            D1r=disc_d1(x1_t,t1,x1_tp1.detach()).view(-1)
            D2r=disc_d2(x2_t,t2,x2_tp1.detach()).view(-1)
            errDr = (nn.functional.softplus(-D1r.float()).mean()+nn.functional.softplus(-D2r.float()).mean())
            errDr.backward(retain_graph=True)
            # D fake
            z1=torch.randn(B,nz,device=device); z2=torch.randn(B,nz,device=device)
            with (gen_diff_1.no_sync(), gen_diff_2.no_sync(), gen_nd_12.no_sync(), gen_nd_21.no_sync()):
                p1=gen_nd_21(r2); p2=gen_nd_12(r1)
                pd1=gen_diff_1(torch.cat((x1_tp1.detach(),p2),1),t1,z1)
                pd2=gen_diff_2(torch.cat((x2_tp1.detach(),p1),1),t2,z2)
                s1=sample_posterior(pos_coeff,pd1[:,[0],:],x1_tp1,t1)
                s2=sample_posterior(pos_coeff,pd2[:,[0],:],x2_tp1,t2)
                o1=disc_d1(s1,t1,x1_tp1.detach()).view(-1)
                o2=disc_d2(s2,t2,x2_tp1.detach()).view(-1)
                errDf=nn.functional.softplus(o1.float()).mean()+nn.functional.softplus(o2.float()).mean()
                errDf.backward()
            opt_d_d1.step(); opt_d_d2.step()

            # --- D cycle ---
            disc_c1.zero_grad(); disc_c2.zero_grad()
            Dc1r=disc_c1(r1).view(-1); Dc2r=disc_c2(r2).view(-1)
            errDcr=(nn.functional.softplus(-Dc1r.float()).mean()+nn.functional.softplus(-Dc2r.float()).mean())
            errDcr.backward(retain_graph=True)
            pc1=gen_nd_21(r2); pc2=gen_nd_12(r1)
            Dc1f=disc_c1(pc1).view(-1); Dc2f=disc_c2(pc2).view(-1)
            errDcf=(nn.functional.softplus(Dc1f.float()).mean()+nn.functional.softplus(Dc2f.float()).mean())
            errDcf.backward()
            opt_d_c1.step(); opt_d_c2.step()

            # --- G phase ---
            gen_diff_1.zero_grad(); gen_diff_2.zero_grad()
            gen_nd_12.zero_grad(); gen_nd_21.zero_grad()
            t1=torch.randint(0,args.num_timesteps,(B,),device=device)
            t2=torch.randint(0,args.num_timesteps,(B,),device=device)
            x1_t,x1_tp1=q_sample_pairs(coeff,r1,t1)
            x2_t,x2_tp1=q_sample_pairs(coeff,r2,t2)
            z1=torch.randn(B,nz,device=device); z2=torch.randn(B,nz,device=device)
            p1=gen_nd_21(r2); p2c=gen_nd_12(p1)
            p2=gen_nd_12(r1); p1c=gen_nd_21(p2)
            pd1=gen_diff_1(torch.cat((x1_tp1.detach(),p2),1),t1,z1)
            pd2=gen_diff_2(torch.cat((x2_tp1.detach(),p1),1),t2,z2)
            s1=sample_posterior(pos_coeff,pd1[:,[0],:],x1_tp1,t1)
            s2=sample_posterior(pos_coeff,pd2[:,[0],:],x2_tp1,t2)
            o1=disc_d1(s1,t1,x1_tp1.detach()).view(-1)
            o2=disc_d2(s2,t2,x2_tp1.detach()).view(-1)
            Dc1f=disc_c1(p1).view(-1); Dc2f=disc_c2(p2).view(-1)
            errG = (nn.functional.softplus(-o1.float()).mean()+nn.functional.softplus(-o2.float()).mean()+
                    nn.functional.softplus(-Dc1f.float()).mean()+nn.functional.softplus(-Dc2f.float()).mean()+
                    10.0*(nn.functional.l1_loss(p1c,r1)+nn.functional.l1_loss(p2c,r2))+
                    10.0*(nn.functional.l1_loss(pd1[:,[0],:],r1)+nn.functional.l1_loss(pd2[:,[0],:],r2)))
            errG.backward()
            opt_g_d1.step(); opt_g_d2.step(); opt_g_nd12.step(); opt_g_nd21.step()

        # === Validation ===
        g1 = gen_diff_1.module if hasattr(gen_diff_1,'module') else gen_diff_1
        nd12 = gen_nd_12.module if hasattr(gen_nd_12,'module') else gen_nd_12
        nd21 = gen_nd_21.module if hasattr(gen_nd_21,'module') else gen_nd_21
        nr1, ss1, nr2, ss2 = [], [], [], []
        for xv, yv in loader_val:
            r1v=xv.to(device); r2v=yv.to(device)
            with torch.no_grad():
                xt1=torch.cat((torch.randn_like(r1v),r2v),1)
                p1v=sample_from_model(pos_coeff,g1,args.num_timesteps,xt1,T,args)
                xt2=torch.cat((torch.randn_like(r2v),r1v),1)
                p2v=sample_from_model(pos_coeff,g1,args.num_timesteps,xt2,T,args) # same gen1 for both dirs (matching train.py)
            p1n=to_range_0_1(p1v).squeeze().cpu().numpy()
            p2n=to_range_0_1(p2v).squeeze().cpu().numpy()
            r1n=to_range_0_1(r1v).squeeze().cpu().numpy()
            r2n=to_range_0_1(r2v).squeeze().cpu().numpy()
            nr1.append(compute_nrmse(p1n,r1n)); ss1.append(compute_ssim_official(p1n[np.newaxis],r1n[np.newaxis]))
            nr2.append(compute_nrmse(p2n,r2n)); ss2.append(compute_ssim_official(p2n[np.newaxis],r2n[np.newaxis]))
        if rank==0:
            print(f'VAL epoch {epoch}: nrmse_d1={np.mean(nr1):.4f} ssim_d1={np.mean(ss1):.4f} nrmse_d2={np.mean(nr2):.4f} ssim_d2={np.mean(ss2):.4f}')
            with open(f'{exp_path}/metrics.csv','a') as f:
                if epoch==1: f.write('epoch,nrmse_d1,ssim_d1,nrmse_d2,ssim_d2\n')
                f.write(f'{epoch},{np.mean(nr1):.4f},{np.mean(ss1):.4f},{np.mean(nr2):.4f},{np.mean(ss2):.4f}\n')

def init_processes(rank, size, fn, args, *extra):
    os.environ['MASTER_ADDR']=args.master_address; os.environ['MASTER_PORT']=args.port_num
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl',init_method='env://',rank=rank,world_size=size)
    fn(rank, rank, args, *extra)
    dist.barrier(); dist.destroy_process_group()

def run_pretrain(rank, gpu, args):
    device = torch.device(f'cuda:{gpu}')
    sd12, sd21 = pretrain_nondiff(args, device, rank)
    if rank==0:
        torch.save(sd12, os.path.join(args.output_path, args.exp, 'nd_12_pretrained.pth'))
        torch.save(sd21, os.path.join(args.output_path, args.exp, 'nd_21_pretrained.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='/data0/syndiff_data/1.5T_to_7T')
    parser.add_argument('--output_path', default='/NAS_writeable/SynDiff/checkpoints')
    parser.add_argument('--exp', default='pretrained_nondiff')
    parser.add_argument('--contrast1', default='1.5T')
    parser.add_argument('--contrast2', default='7T')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--pretrain_epochs', type=int, default=5)
    parser.add_argument('--lr_g', type=float, default=1.6e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_process_per_node', type=int, default=2)
    parser.add_argument('--master_address', default='127.0.0.1')
    parser.add_argument('--port_num', default='6650')
    # NCSNpp args
    for a in ['image_size=256','num_channels=2','num_channels_dae=64','ch_mult=(1,1,2,2,4,4)','num_res_blocks=2',
              'num_timesteps=4','attn_resolutions=(16,)','dropout=0.','resamp_with_conv=True','conditional=True',
              'fir=True','fir_kernel=[1,3,3,1]','skip_rescale=True','resblock_type=biggan','progressive=none',
              'progressive_input=residual','progressive_combine=sum','embedding_type=positional','fourier_scale=16.',
              'not_use_tanh=False','z_emb_dim=256','t_emb_dim=256','ngf=64','nz=100','n_mlp=3','centered=True',
              'beta_min=0.1','beta_max=20.','use_geometric=False','use_ema=True','ema_decay=0.999',
              'r1_gamma=1.0','lazy_reg=10','lambda_l1_loss=10.0','no_lr_decay=True','save_content=True',
              'save_content_every=1','save_ckpt_every=1','local_rank=0']:
        k,v = a.split('=')
        parser.add_argument(f'--{k}', default=v)
    args = parser.parse_args()
    # Parse tuple/list args
    for a in ['ch_mult','attn_resolutions','fir_kernel']:
        v = getattr(args, a)
        if isinstance(v, str): setattr(args, a, eval(v))
    for a in ['resamp_with_conv','conditional','fir','skip_rescale','not_use_tanh','centered','use_geometric','use_ema','save_content','no_lr_decay']:
        v = getattr(args, a)
        if isinstance(v, str): setattr(args, a, v.lower() in ('true','1'))
    args.world_size = args.num_process_per_node
    args.local_rank = 0

    # Phase 1: Pre-train on rank 0 only (single GPU, no DDP needed for pre-train)
    # Actually, pre-train across both GPUs with DDP
    os.makedirs(os.path.join(args.output_path, args.exp), exist_ok=True)
    processes = []
    for rank in range(args.num_process_per_node):
        pa = copy.deepcopy(args); pa.local_rank = rank
        p = Process(target=init_processes, args=(rank, args.world_size, run_pretrain, pa))
        p.start(); processes.append(p)
    for p in processes: p.join()

    # Phase 2: Full SynDiff with pre-trained weights
    sd12 = torch.load(os.path.join(args.output_path, args.exp, 'nd_12_pretrained.pth'), map_location='cpu')
    sd21 = torch.load(os.path.join(args.output_path, args.exp, 'nd_21_pretrained.pth'), map_location='cpu')
    processes = []
    for rank in range(args.num_process_per_node):
        pa = copy.deepcopy(args); pa.local_rank = rank
        p = Process(target=init_processes, args=(rank, args.world_size, train_syndiff_pretrained, pa, sd12, sd21))
        p.start(); processes.append(p)
    for p in processes: p.join()
