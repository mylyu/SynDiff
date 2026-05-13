#!/usr/bin/env python3
"""Experiment B: Baseline CUT + SynDiff .mat data.

Trains the MRIxFields2026 CUT model using our .mat data pipeline.
Uses baseline CUT hyperparameters (lr=2e-4, batch=8, GAN+PatNCE loss).
"""
import argparse, os, sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'experiments'))

from exp_b_adapter_zmap import CUTMatDataAdapterZMap as CUTMatDataAdapter
from cut_model import CUTModel
from unpaired_loader import UnpairedDataLoader


def main():
    parser = argparse.ArgumentParser("Experiment B: CUT + .mat data")
    parser.add_argument('--input_path', required=True, help='Path to .mat data directory')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--exp', required=True)
    parser.add_argument('--src_field', default='1.5T')
    parser.add_argument('--tgt_field', default='7T')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = os.path.join(args.output_dir, args.exp)
    os.makedirs(output_path, exist_ok=True)
    weights_dir = os.path.join(output_path, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    print(f"Experiment B: CUT + .mat data, {args.src_field}->{args.tgt_field}")
    print(f"Data: {args.input_path}, Output: {output_path}")

    # Build CUT model with baseline hyperparams
    model = CUTModel(
        input_nc=2, output_nc=1, ngf=64, ndf=64, n_blocks=9,
        nce_layers=[0, 4, 8, 12, 16], nce_T=0.07, num_patches=256,
        lambda_GAN=1.0, lambda_NCE=1.0, nce_idt=True,
        gan_mode="lsgan", netF_nc=256,
        lr=args.lr, beta1=0.5, beta2=0.999,
    ).to(device)
    model.init_weights()

    # Create datasets using our .mat adapter
    dataset_src = CUTMatDataAdapter(args.input_path, args.src_field, "train")
    dataset_tgt = CUTMatDataAdapter(args.input_path, args.tgt_field, "train")
    print(f"Source dataset ({args.src_field}): {len(dataset_src)} slices")
    print(f"Target dataset ({args.tgt_field}): {len(dataset_tgt)} slices")

    # Wrap in baseline's UnpairedDataLoader
    loader = UnpairedDataLoader(
        dataset_src, dataset_tgt,
        batch_size=args.batch_size, num_workers=4,
    )

    # Data-dependent init (required by CUT for PatchSampleF MLPs)
    print("Running data-dependent initialization...")
    for batch_a, batch_b in loader:
        init_batch_a = batch_a["image"].to(device)
        init_batch_b = batch_b["image"].to(device)
        break
    model.data_dependent_initialize(init_batch_a, init_batch_b)
    model.setup_optimizers()

    # LR schedulers (linear decay after n_epochs — baseline uses 0 decay)
    def get_lr_lambda(n_epochs, n_epochs_decay):
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
        return lambda_rule
    lr_lambda = get_lr_lambda(args.num_epoch, 0)
    scheduler_G = optim.lr_scheduler.LambdaLR(model.optimizer_G, lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(model.optimizer_D, lr_lambda)
    scheduler_F = optim.lr_scheduler.LambdaLR(model.optimizer_F, lr_lambda)

    # Training loop
    print(f"Starting {args.num_epoch} epochs...")
    metrics_csv = os.path.join(output_path, 'metrics.csv')
    with open(metrics_csv, 'w') as f:
        f.write('epoch,loss_G,loss_D,loss_NCE\n')

    for epoch in range(args.num_epoch):
        model.train()
        epoch_losses = {'loss_G': 0.0, 'loss_D': 0.0, 'loss_NCE': 0.0}
        n_batches = 0

        for i, (batch_a, batch_b) in enumerate(loader):
            real_A = batch_a["image"].to(device)
            real_B = batch_b["image"].to(device)
            losses = model.optimize_parameters(real_A, real_B)
            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k]
            n_batches += 1

        scheduler_G.step()
        scheduler_D.step()
        scheduler_F.step()

        avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        print(f"[{epoch+1}/{args.num_epoch}] G={avg_losses['loss_G']:.4f} "
              f"D={avg_losses['loss_D']:.4f} NCE={avg_losses.get('loss_NCE', 0):.4f}")

        with open(metrics_csv, 'a') as f:
            f.write(f'{epoch+1},{avg_losses["loss_G"]:.4f},{avg_losses["loss_D"]:.4f},{avg_losses.get("loss_NCE", 0):.4f}\n')

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(weights_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer_G": model.optimizer_G.state_dict(),
                "optimizer_D": model.optimizer_D.state_dict(),
                "optimizer_F": model.optimizer_F.state_dict(),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    print(f"Experiment B complete. Output: {output_path}")


if __name__ == '__main__':
    main()
