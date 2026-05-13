#!/usr/bin/env python3
"""Evaluate SynDiff non-diffusive generators (ResnetGenerator, 6 blocks) directly.

Runs single-pass translation (no diffusion) and computes metrics.
Answers: is the non-diffusive module already failing before the diffusive module sees it?

Usage:
    python3 experiments/eval_nondiff.py --exp syn_4090Dx2_01T_7T_v2 --epochs 1,3,5,10,20 --gpu 0
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
import backbones.generator_resnet
from dataset import CreateDatasetSynthesis

def nrmse(pred, target):
    d = pred.astype(np.float64) - target.astype(np.float64)
    n = np.linalg.norm(target.astype(np.float64))
    return float(np.linalg.norm(d)/n) if n > 1e-10 else 0.0

def ssim_official(pred, target):
    vals = []
    for p, t in zip(pred, target):
        dr = t.max() - t.min()
        vals.append(1.0 if dr < 1e-10 else ssim(t, p, data_range=dr))
    return float(np.mean(vals))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='syn_4090Dx2_01T_7T_v2')
    parser.add_argument('--input_path', default='/data0/syndiff_data/0.1T_to_7T')
    parser.add_argument('--output_path', default='/NAS_writeable/SynDiff/checkpoints')
    parser.add_argument('--epochs', type=str, default='1,3,5,10,20')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.gpu)
    epochs = [int(x) for x in args.epochs.split(',')]

    dataset = CreateDatasetSynthesis('test', args.input_path, '0.1T', '7T')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    to_range_0_1 = lambda x: (x + 1.) / 2.
    crop = transforms.CenterCrop((256, 256))

    exp_path = os.path.join(args.output_path, args.exp)
    out_dir = os.path.join(exp_path, 'nondiff_eval')
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for ep in epochs:
        print(f"\n=== Epoch {ep} ===")
        nd_12 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[args.gpu]).to(device)
        nd_21 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[args.gpu]).to(device)

        ckpt_12 = f'{exp_path}/gen_non_diffusive_1to2_{ep}.pth'
        ckpt_21 = f'{exp_path}/gen_non_diffusive_2to1_{ep}.pth'
        if not os.path.exists(ckpt_12):
            print(f"  Checkpoint not found, skipping")
            continue
        # Strip DDP/DataParallel wrapper prefixes from saved checkpoint
        ckpt12_raw = torch.load(ckpt_12, map_location=device)
        ckpt12_clean = {k.replace('module.module.', 'module.').replace('module.', ''): v
                        for k, v in ckpt12_raw.items()}
        nd_12.load_state_dict(ckpt12_clean, strict=False)
        ckpt21_raw = torch.load(ckpt_21, map_location=device)
        ckpt21_clean = {k.replace('module.module.', 'module.').replace('module.', ''): v
                        for k, v in ckpt21_raw.items()}
        nd_21.load_state_dict(ckpt21_clean, strict=False)
        nd_12.eval(); nd_21.eval()

        ep_dir = os.path.join(out_dir, f'epoch_{ep}')
        os.makedirs(ep_dir, exist_ok=True)

        for dname, gen, swap in [('dir1_1to2', nd_12, False), ('dir2_2to1', nd_21, True)]:
            psnr_tgt, ssim_tgt, nrmse_tgt, mae_tgt = [], [], [], []
            psnr_src, ssim_src, nrmse_src, mae_src = [], [], [], []

            for i, (x, y) in enumerate(data_loader):
                if swap:
                    real_data, source_data = y.to(device), x.to(device)
                else:
                    real_data, source_data = x.to(device), y.to(device)

                with torch.no_grad():
                    pred = gen(source_data)

                pred_np = crop(to_range_0_1(pred)).squeeze().cpu().numpy()
                src_np = crop(to_range_0_1(source_data)).squeeze().cpu().numpy()
                tgt_np = crop(to_range_0_1(real_data)).squeeze().cpu().numpy()

                psnr_tgt.append(psnr_skim(tgt_np, pred_np, data_range=1.0))
                ssim_tgt.append(ssim_official(pred_np[np.newaxis], tgt_np[np.newaxis]))
                nrmse_tgt.append(nrmse(pred_np, tgt_np))
                mae_tgt.append(np.mean(np.abs(pred_np - tgt_np)))

                psnr_src.append(psnr_skim(src_np, pred_np, data_range=1.0))
                ssim_src.append(ssim_official(pred_np[np.newaxis], src_np[np.newaxis]))
                nrmse_src.append(nrmse(pred_np, src_np))
                mae_src.append(np.mean(np.abs(pred_np - src_np)))

                if i < 10:
                    panel = torch.cat([torch.from_numpy(src_np).unsqueeze(0),
                                       torch.from_numpy(pred_np).unsqueeze(0),
                                       torch.from_numpy(tgt_np).unsqueeze(0),
                                       torch.from_numpy(np.abs(pred_np - tgt_np)).unsqueeze(0)], dim=-1)
                    torchvision.utils.save_image(panel, f'{ep_dir}/{dname}_case{i:02d}.png', normalize=True)

            src_closer = np.mean(psnr_src) > np.mean(psnr_tgt)
            print(f"  {dname}: PSNR_tgt={np.mean(psnr_tgt):.2f} SSIM_tgt={np.mean(ssim_tgt):.4f} "
                  f"nRMSE_tgt={np.mean(nrmse_tgt):.4f} PSNR_src={np.mean(psnr_src):.2f} "
                  f"{'*** SOURCE BIAS ***' if src_closer else ''}")
            results[f'{dname}_ep{ep}'] = {
                'psnr_tgt': (np.mean(psnr_tgt), np.std(psnr_tgt)),
                'nrmse_tgt': (np.mean(nrmse_tgt), np.std(nrmse_tgt)),
                'ssim_tgt': (np.mean(ssim_tgt), np.std(ssim_tgt)),
                'psnr_src': (np.mean(psnr_src), np.std(psnr_src)),
                'source_bias': src_closer,
            }

    print("\n=== NON-DIFFUSIVE SUMMARY ===")
    print(f"{'Dir/Epoch':<15} {'nRMSE_tgt':>10} {'SSIM_tgt':>10} {'PSNR_tgt':>10} {'PSNR_src':>10} {'Bias?':>6}")
    for k, v in sorted(results.items()):
        print(f"{k:<15} {v['nrmse_tgt'][0]:10.4f} {v['ssim_tgt'][0]:10.4f} "
              f"{v['psnr_tgt'][0]:10.2f} {v['psnr_src'][0]:10.2f} {'YES' if v.get('source_bias') else '':>6}")

    np.savez(os.path.join(out_dir, 'results.npz'), results=results)
    print(f"\nResults saved to {out_dir}")

if __name__ == '__main__':
    main()
