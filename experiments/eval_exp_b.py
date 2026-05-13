#!/usr/bin/env python3
"""Evaluate Exp B CUT checkpoint on test set. Computes nRMSE and SSIM_official."""
import argparse, os, sys
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'experiments'))

from exp_b_adapter import CUTMatDataAdapter
from cut_model import CUTModel
from networks import ResnetGenerator

def nrmse(pred, target):
    d = pred.astype(np.float64) - target.astype(np.float64)
    n = np.linalg.norm(target.astype(np.float64))
    return float(np.linalg.norm(d)/n) if n > 1e-10 else 0.0

def ssim_official(pred, target):
    vals = []
    for p, t in zip(pred, target):
        dr = t.max() - t.min()
        vals.append(1.0 if dr<1e-10 else ssim(t, p, data_range=dr))
    return float(np.mean(vals))

def to_range_0_1(x):
    return (x + 1.) / 2.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='/data0/syndiff_data/1.5T_to_7T')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load generator only (discriminator and netF not needed for inference)
    netG = ResnetGenerator(input_nc=1, output_nc=1, ngf=64, n_blocks=9).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    # Extract generator weights from full model state dict
    model_state = ckpt['model']
    gen_state = {k.replace('netG.', ''): v for k, v in model_state.items() if k.startswith('netG.')}
    netG.load_state_dict(gen_state)
    netG.eval()

    # Load test data
    dataset_src = CUTMatDataAdapter(args.input_path, '1.5T', 'test')
    dataset_tgt = CUTMatDataAdapter(args.input_path, '7T', 'test')

    nrmse_vals = []
    ssim_vals = []
    with torch.no_grad():
        for i in range(len(dataset_src)):
            src = dataset_src[i]['image'].unsqueeze(0).to(device)  # (1,1,H,W)
            tgt = dataset_tgt[i]['image'].unsqueeze(0).to(device)

            fake = netG(src)

            fake = to_range_0_1(fake).squeeze().cpu().numpy()
            tgt_np = to_range_0_1(tgt).squeeze().cpu().numpy()

            nrmse_vals.append(nrmse(fake, tgt_np))
            ssim_vals.append(ssim_official(fake[np.newaxis], tgt_np[np.newaxis]))

    print(f"Exp B CUT Results ({os.path.basename(args.ckpt)}):")
    print(f"  nRMSE: {np.mean(nrmse_vals):.4f} ± {np.std(nrmse_vals):.4f}")
    print(f"  SSIM:  {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}")

if __name__ == '__main__':
    main()
