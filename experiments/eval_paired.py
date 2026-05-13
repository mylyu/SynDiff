#!/usr/bin/env python3
"""Paired evaluation of SynDiff checkpoints on test data.

Computes pred-vs-target AND pred-vs-source metrics to detect source bias.
Saves qualitative panels (source, target, pred, abs-error).

Usage:
    python3 experiments/eval_paired.py --exp syn_V100x8_15T_7T_v2 --epochs 1,3,5,10,20 --gpu 0
"""
import argparse, os, sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_skim

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from backbones.ncsnpp_generator_adagn import NCSNpp
from dataset import CreateDatasetSynthesis

# ---- Diffusion helpers (same as test.py) ----
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    return 1. - torch.exp(2. * log_mean_coeff)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    t = np.arange(0, n_timestep + 1, dtype=np.float64) / n_timestep
    t = torch.from_numpy(t) * (1. - 1e-3) + 1e-3
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min, beta_max = args.beta_min, args.beta_max
    t = np.arange(0, n_timestep + 1, dtype=np.float64) / n_timestep
    t = torch.from_numpy(t) * (1. - 1e-3) + 1e-3
    var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device).float()
    return betas**0.5, torch.sqrt(1 - betas), betas

class Posterior_Coefficients:
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.], device=device), self.alphas_cumprod[:-1]), 0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def sample_posterior(coefficients, x_0, x_t, t):
    mean = (extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t)
    log_var = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
    noise = torch.randn_like(x_t)
    nonzero_mask = (1 - (t == 0).float())
    return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

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

def load_checkpoint(filepath, netG, device='cuda:0'):
    ckpt = torch.load(filepath, map_location=device)
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

# ---- Metrics ----
def nrmse(pred, target):
    d = pred.astype(np.float64) - target.astype(np.float64)
    n = np.linalg.norm(target.astype(np.float64))
    return float(np.linalg.norm(d)/n) if n > 1e-10 else 0.0

def ssim_official(pred, target):
    from skimage.metrics import structural_similarity as _ssim
    vals = []
    for p, t in zip(pred, target):
        dr = t.max() - t.min()
        vals.append(1.0 if dr < 1e-10 else _ssim(t, p, data_range=dr))
    return float(np.mean(vals))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='syn_4090Dx2_01T_7T_v2')
    parser.add_argument('--input_path', default='/data0/syndiff_data/0.1T_to_7T')
    parser.add_argument('--output_path', default='/NAS_writeable/SynDiff/checkpoints')
    parser.add_argument('--epochs', type=str, default='1,3,5,10,20')
    parser.add_argument('--gpu', type=int, default=0)
    # NCSNpp args
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels_dae', type=int, default=64)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1,1,2,2,4,4])
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
    parser.add_argument('--num_channels', type=int, default=2)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--centered', action='store_false', default=True)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.gpu)
    epochs = [int(x) for x in args.epochs.split(',')]

    dataset = CreateDatasetSynthesis('test', args.input_path, '0.1T', '7T')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    to_range_0_1 = lambda x: (x + 1.) / 2.
    crop = transforms.CenterCrop((256, 256))

    exp_path = os.path.join(args.output_path, args.exp)
    out_dir = os.path.join(exp_path, 'paired_eval')
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for ep in epochs:
        print(f"\n=== Epoch {ep} ===")
        gen1 = NCSNpp(args).to(device)
        gen2 = NCSNpp(args).to(device)
        ckpt1 = f'{exp_path}/gen_diffusive_1_{ep}.pth'
        ckpt2 = f'{exp_path}/gen_diffusive_2_{ep}.pth'
        if not os.path.exists(ckpt1):
            print(f"  Checkpoint not found: {ckpt1}, skipping")
            continue
        load_checkpoint(ckpt1, gen1, device)
        load_checkpoint(ckpt2, gen2, device)

        ep_dir = os.path.join(out_dir, f'epoch_{ep}')
        os.makedirs(ep_dir, exist_ok=True)

        for dname, gen, swap in [('dir1', gen1, False), ('dir2', gen2, True)]:
            results_list = {'psnr_tgt': [], 'ssim_tgt': [], 'nrmse_tgt': [], 'mae_tgt': [],
                           'psnr_src': [], 'ssim_src': [], 'nrmse_src': [], 'mae_src': []}
            all_preds, all_srcs, all_tgts = [], [], []

            for i, (x, y) in enumerate(data_loader):
                if swap:
                    real_data, source_data = y.to(device), x.to(device)
                else:
                    real_data, source_data = x.to(device), y.to(device)

                x_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
                pred = sample_from_model(pos_coeff, gen, args.num_timesteps, x_t, T, args)

                pred_np = crop(to_range_0_1(pred)).squeeze().cpu().numpy()
                src_np = crop(to_range_0_1(source_data)).squeeze().cpu().numpy()
                tgt_np = crop(to_range_0_1(real_data)).squeeze().cpu().numpy()

                # pred vs target
                results_list['psnr_tgt'].append(psnr_skim(tgt_np, pred_np, data_range=1.0))
                results_list['ssim_tgt'].append(ssim_official(pred_np[np.newaxis], tgt_np[np.newaxis]))
                results_list['nrmse_tgt'].append(nrmse(pred_np, tgt_np))
                results_list['mae_tgt'].append(np.mean(np.abs(pred_np - tgt_np)))

                # pred vs source (source-similarity — detects identity bias)
                results_list['psnr_src'].append(psnr_skim(src_np, pred_np, data_range=1.0))
                results_list['ssim_src'].append(ssim_official(pred_np[np.newaxis], src_np[np.newaxis]))
                results_list['nrmse_src'].append(nrmse(pred_np, src_np))
                results_list['mae_src'].append(np.mean(np.abs(pred_np - src_np)))

                # Save first 10 cases as panels
                if i < 10:
                    panel = torch.cat([torch.from_numpy(src_np).unsqueeze(0),
                                       torch.from_numpy(pred_np).unsqueeze(0),
                                       torch.from_numpy(tgt_np).unsqueeze(0),
                                       torch.from_numpy(np.abs(pred_np - tgt_np)).unsqueeze(0)], dim=-1)
                    torchvision.utils.save_image(panel, f'{ep_dir}/{dname}_case{i:02d}.png', normalize=True)
                all_preds.append(pred_np); all_srcs.append(src_np); all_tgts.append(tgt_np)

            # Print results
            src_closer = np.mean(results_list['psnr_src']) > np.mean(results_list['psnr_tgt'])
            print(f"  {dname}: PSNR_tgt={np.mean(results_list['psnr_tgt']):.2f} "
                  f"SSIM_tgt={np.mean(results_list['ssim_tgt']):.4f} "
                  f"nRMSE_tgt={np.mean(results_list['nrmse_tgt']):.4f} "
                  f"PSNR_src={np.mean(results_list['psnr_src']):.2f} "
                  f"{'*** SOURCE BIAS ***' if src_closer else ''}")
            results[f'{dname}_ep{ep}'] = {k: (np.mean(v), np.std(v)) for k, v in results_list.items()}
            results[f'{dname}_ep{ep}']['source_bias'] = src_closer

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'Dir/Epoch':<15} {'nRMSE_tgt':>10} {'SSIM_tgt':>10} {'PSNR_tgt':>10} {'PSNR_src':>10} {'Bias?':>6}")
    for k, v in sorted(results.items()):
        print(f"{k:<15} {v['nrmse_tgt'][0]:10.4f} {v['ssim_tgt'][0]:10.4f} "
              f"{v['psnr_tgt'][0]:10.2f} {v['psnr_src'][0]:10.2f} {'YES' if v.get('source_bias') else '':>6}")

    np.savez(os.path.join(out_dir, 'results.npz'), results=results)
    print(f"\nResults saved to {out_dir}")

if __name__ == '__main__':
    main()
