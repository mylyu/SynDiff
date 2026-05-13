# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SynDiff is a PyTorch implementation of "Unsupervised Medical Image Translation With Adversarial Diffusion Models" (IEEE TMI 2023). The project implements adversarial diffusion models for translating between medical image modalities (e.g., T1 ↔ T2, T1 ↔ PD) without requiring paired training data.

## Dependencies

- Python ≥ 3.6.9
- PyTorch ≥ 1.7.1
- torchvision ≥ 0.8.2
- CUDA ≥ 11.2
- ninja
- python3.x-dev (apt install, matching your Python version)

## Dataset Structure

Datasets must be structured as `.mat` files with shape `(#images, width, height)` and values between 0-1.0:

```
input_path/
  ├── data_train_contrast1.mat
  ├── data_train_contrast2.mat
  ├── data_val_contrast1.mat
  ├── data_val_contrast2.mat
  ├── data_test_contrast1.mat
  └── data_test_contrast2.mat
```

The dataset loader (`dataset.py`) automatically pads images to 256x256 and normalizes to [-1, 1] range.

## Training

Training supports multi-GPU distributed training via PyTorch DDP:

```bash
python3 train.py \
  --image_size 256 \
  --exp exp_syndiff \
  --num_channels 2 \
  --num_channels_dae 64 \
  --ch_mult 1 1 2 2 4 4 \
  --num_timesteps 4 \
  --num_res_blocks 2 \
  --batch_size 1 \
  --contrast1 T1 \
  --contrast2 T2 \
  --num_epoch 500 \
  --ngf 64 \
  --embedding_type positional \
  --use_ema \
  --ema_decay 0.999 \
  --r1_gamma 1. \
  --z_emb_dim 256 \
  --lr_d 1e-4 \
  --lr_g 1.6e-4 \
  --lazy_reg 10 \
  --num_process_per_node 1 \
  --save_content \
  --local_rank 0 \
  --input_path /input/path/for/data \
  --output_path /output/for/results
```

**Key Training Parameters:**
- `--num_timesteps`: Number of diffusion steps (default 4)
- `--num_process_per_node`: Number of GPUs for DDP
- `--lazy_reg`: Apply gradient penalty every N steps (None = every step)
- `--use_ema`: Enable exponential moving average for generator weights
- `--save_content_every` / `--save_ckpt_every`: Checkpoint frequency

## Testing

Inference loads two pre-trained generator checkpoints (one per translation direction):

```bash
python test.py \
  --image_size 256 \
  --exp exp_syndiff \
  --num_channels 2 \
  --num_channels_dae 64 \
  --ch_mult 1 1 2 2 4 4 \
  --num_timesteps 4 \
  --num_res_blocks 2 \
  --batch_size 1 \
  --embedding_type positional \
  --z_emb_dim 256 \
  --contrast1 T1 \
  --contrast2 T2 \
  --which_epoch 50 \
  --gpu_chose 0 \
  --input_path /input/path/for/data \
  --output_path /output/for/results
```

Outputs are saved to `{output_path}/{exp}/generated_samples/epoch_{which_epoch}/` including PSNR values (`.npy`) and synthetic images (`.mat`).

## Architecture

**Dual-Generator System:**
1. **Diffusive Generators** (`backbones/ncsnpp_generator_adagn.py`): NCSN++ architecture with adaptive group normalization
   - `gen_diffusive_1`: Contrast1 → Contrast2
   - `gen_diffusive_2`: Contrast2 → Contrast1
   
2. **Non-Diffusive Generators** (`backbones/generator_resnet.py`): ResNet-6 blocks
   - `gen_non_diffusive_1to2`: Contrast1 → Contrast2 (translation)
   - `gen_non_diffusive_2to1`: Contrast2 → Contrast1 (translation)

**Discriminators:**
- `disc_diffusive_1` / `disc_diffusive_2` (`backbones/discriminator.py`): Large discriminators with time embedding for diffusive GAN training
- `disc_non_diffusive_cycle1` / `disc_non_diffusive_cycle2`: For cycle-consistency loss

**Diffusion Process:**
- Variance schedule: VP (variance preserving) by default, geometric optional via `--use_geometric`
- Beta range: `beta_min=0.1`, `beta_max=20.` (configurable)
- Posterior sampling computed analytically for denoising

**Loss Components:**
- Adversarial loss (GAN)
- L1 reconstruction loss (`--lambda_l1_loss`)
- Cycle-consistency loss
- R1 gradient penalty on discriminators

## Multi-GPU Training

For multi-GPU training, set:
```bash
--num_process_per_node <num_gpus>
--master_address 127.0.0.1
--port_num 6021
```

Each GPU runs a separate process with `torch.multiprocessing.Process`. Synchronization uses NCCL backend.

## Checkpoint Structure

Checkpoints (`content.pth`) contain full training state for resuming:
- Generator/discriminator state dicts (both diffusive and non-diffusive)
- Optimizer and scheduler states
- Training metadata (epoch, global_step, args)

Individual generator checkpoints are saved as `{name}_{epoch}.pth` with EMA-swapped weights if `--use_ema` is enabled.

## Sample Data

Sample toy data is available in `SynDiff_sample_data/` containing `T1.mat` and `T2.mat` for testing the pipeline.
