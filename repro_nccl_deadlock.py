#!/usr/bin/env python3
"""Minimal reproduction of NCCL deadlock on V100.

Uses a tiny 32x32 image, 2 GPUs, same NCSNpp+Discriminator models as train.py.
Runs N training iterations, then calls dist.barrier() and dist.destroy_process_group()
to find which NCCL collective hangs.

Usage:
    python3 repro_nccl_deadlock.py --gpus 0,1 --iters 10
    python3 repro_nccl_deadlock.py --gpus 0,1,2,3 --iters 50  # test 4 GPUs
"""

import argparse, os, sys, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process
import numpy as np

def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def log(rank, msg):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_test(rank, gpu, args):
    torch.manual_seed(42 + rank)
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(gpu)

    # Quick imports (inside process for clean DDP init)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
    from backbones.ncsnpp_generator_adagn import NCSNpp
    from backbones.discriminator import Discriminator_large
    import backbones.generator_resnet

    log(rank, f"Creating models...")

    args.num_channels = 2
    gen1 = NCSNpp(args).to(device)
    gen2 = NCSNpp(args).to(device)
    args.num_channels = 1
    gen_nd_12 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[gpu])
    gen_nd_21 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[gpu])

    disc1 = Discriminator_large(nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim,
                                  act=nn.LeakyReLU(0.2)).to(device)
    disc2 = Discriminator_large(nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim,
                                  act=nn.LeakyReLU(0.2)).to(device)
    disc_c1 = backbones.generator_resnet.define_D(gpu_ids=[gpu])
    disc_c2 = backbones.generator_resnet.define_D(gpu_ids=[gpu])

    broadcast_params(gen1.parameters()); broadcast_params(gen2.parameters())
    broadcast_params(gen_nd_12.parameters()); broadcast_params(gen_nd_21.parameters())
    broadcast_params(disc1.parameters()); broadcast_params(disc2.parameters())
    broadcast_params(disc_c1.parameters()); broadcast_params(disc_c2.parameters())

    gen1 = nn.parallel.DistributedDataParallel(gen1, device_ids=[gpu], find_unused_parameters=True)
    gen2 = nn.parallel.DistributedDataParallel(gen2, device_ids=[gpu], find_unused_parameters=True)
    gen_nd_12 = nn.parallel.DistributedDataParallel(gen_nd_12, device_ids=[gpu], find_unused_parameters=True)
    gen_nd_21 = nn.parallel.DistributedDataParallel(gen_nd_21, device_ids=[gpu], find_unused_parameters=True)
    disc1 = nn.parallel.DistributedDataParallel(disc1, device_ids=[gpu], find_unused_parameters=True)
    disc2 = nn.parallel.DistributedDataParallel(disc2, device_ids=[gpu], find_unused_parameters=True)
    disc_c1 = nn.parallel.DistributedDataParallel(disc_c1, device_ids=[gpu], find_unused_parameters=True)
    disc_c2 = nn.parallel.DistributedDataParallel(disc_c2, device_ids=[gpu], find_unused_parameters=True)

    log(rank, f"Models ready. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Dummy data
    S = args.image_size
    B = args.batch_size
    x1 = torch.randn(B, 1, S, S, device=device)
    x2 = torch.randn(B, 1, S, S, device=device)
    nz = args.nz

    for iteration in range(1, args.iters + 1):
        t1 = torch.randint(0, args.num_timesteps, (B,), device=device)
        t2 = torch.randint(0, args.num_timesteps, (B,), device=device)

        # D real (with retain_graph=True + R1 gradient penalty — likely deadlock trigger)
        disc1.zero_grad(); disc2.zero_grad()
        x1_t = torch.randn_like(x1); x2_t = torch.randn_like(x2)
        x1_t.requires_grad = True; x2_t.requires_grad = True
        D1_real = disc1(x1_t, t1, x1_t.detach()).view(-1)
        D2_real = disc2(x2_t, t2, x2_t.detach()).view(-1)
        errD_real = (F.softplus(-D1_real).mean() + F.softplus(-D2_real).mean())
        errD_real.backward(retain_graph=True)
        # R1 gradient penalty (double backward)
        grad1 = torch.autograd.grad(outputs=D1_real.sum(), inputs=x1_t, create_graph=True)[0]
        grad2 = torch.autograd.grad(outputs=D2_real.sum(), inputs=x2_t, create_graph=True)[0]
        gp = args.r1_gamma / 2 * (grad1.pow(2).mean() + grad2.pow(2).mean())
        gp.backward()

        # D fake
        with gen1.no_sync(), gen2.no_sync(), gen_nd_12.no_sync(), gen_nd_21.no_sync():
            z1 = torch.randn(B, nz, device=device)
            z2 = torch.randn(B, nz, device=device)
            pred_1 = gen_nd_21(x2)
            pred_2 = gen_nd_12(x1)
            pred_diff1 = gen1(torch.cat((x1_t.detach(), pred_2), dim=1), t1, z1)
            pred_diff2 = gen2(torch.cat((x2_t.detach(), pred_1), dim=1), t2, z2)
            out1 = disc1(pred_diff1[:, [0], :], t1, x1_t.detach()).view(-1)
            out2 = disc2(pred_diff2[:, [0], :], t2, x2_t.detach()).view(-1)
            errD_fake = F.softplus(out1).mean() + F.softplus(out2).mean()
            errD_fake.backward()
        # D step
        disc1.zero_grad(); disc2.zero_grad()

        # D cycle
        disc_c1.zero_grad(); disc_c2.zero_grad()
        D_c1_real = disc_c1(x1).view(-1); D_c2_real = disc_c2(x2).view(-1)
        errD_c_real = F.softplus(-D_c1_real).mean() + F.softplus(-D_c2_real).mean()
        errD_c_real.backward(retain_graph=True)
        pred_c1 = gen_nd_21(x2); pred_c2 = gen_nd_12(x1)
        D_c1_fake = disc_c1(pred_c1).view(-1); D_c2_fake = disc_c2(pred_c2).view(-1)
        errD_c_fake = F.softplus(D_c1_fake).mean() + F.softplus(D_c2_fake).mean()
        errD_c_fake.backward()
        disc_c1.zero_grad(); disc_c2.zero_grad()

        # G
        gen1.zero_grad(); gen2.zero_grad()
        gen_nd_12.zero_grad(); gen_nd_21.zero_grad()
        pred_1 = gen_nd_21(x2); pred_2 = gen_nd_12(x1)
        pred_1_cycle = gen_nd_21(pred_2); pred_2_cycle = gen_nd_12(pred_1)
        pred_diff1 = gen1(torch.cat((x1_t.detach(), pred_2), dim=1), t1, z1)
        pred_diff2 = gen2(torch.cat((x2_t.detach(), pred_1), dim=1), t2, z2)
        out1 = disc1(pred_diff1[:, [0], :], t1, x1_t.detach()).view(-1)
        out2 = disc2(pred_diff2[:, [0], :], t2, x2_t.detach()).view(-1)
        D_c1 = disc_c1(pred_1).view(-1); D_c2 = disc_c2(pred_2).view(-1)
        errG = (F.softplus(-out1).mean() + F.softplus(-out2).mean() +
                F.softplus(-D_c1).mean() + F.softplus(-D_c2).mean() +
                args.lambda_l1 * (F.l1_loss(pred_diff1[:, [0], :], x1) +
                                  F.l1_loss(pred_diff2[:, [0], :], x2) +
                                  F.l1_loss(pred_1_cycle, x1) +
                                  F.l1_loss(pred_2_cycle, x2)))
        errG.backward()
        gen1.zero_grad(); gen2.zero_grad()
        gen_nd_12.zero_grad(); gen_nd_21.zero_grad()

    log(rank, f"Completed {args.iters} training iterations")

    # TEST 1: barrier
    log(rank, f"Test 1: dist.barrier() ...")
    dist.barrier()
    log(rank, f"Test 1: barrier OK")

    # TEST 2: another barrier
    log(rank, f"Test 2: dist.barrier() ...")
    dist.barrier()
    log(rank, f"Test 2: barrier OK")

    # TEST 3: destroy
    log(rank, f"Test 3: dist.destroy_process_group() ...")
    dist.destroy_process_group()
    log(rank, f"Test 3: destroy OK — ALL TESTS PASSED")

def init_processes(rank, size, fn, args, local_rank):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(local_rank)
    gpu = local_rank
    dist_backend = os.environ.get('DIST_BACKEND', 'nccl')
    dist.init_process_group(backend=dist_backend, init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--port', type=str, default='9999')

    # NCSNpp args (minimal, for 32x32)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_channels_dae', type=int, default=64)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1,1,2,2])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--num_timesteps', type=int, default=4)
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
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--centered', action='store_false', default=True)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--lambda_l1', type=float, default=0.5)
    parser.add_argument('--r1_gamma', type=float, default=1.0)

    args = parser.parse_args()
    gpu_list = [int(x) for x in args.gpus.split(',')]
    size = len(gpu_list)
    args.world_size = size
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.master_address = '127.0.0.1'
    args.port_num = args.port

    print(f"=== NCCL deadlock repro: {size} GPUs, {args.iters} iters, 32x32 images ===")
    t0 = time.time()

    processes = []
    for rank in range(size):
        pa = copy.deepcopy(args)
        pa.local_rank = rank
        p = Process(target=init_processes, args=(rank, size, run_test, pa, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print(f"Total: {time.time() - t0:.0f}s")
