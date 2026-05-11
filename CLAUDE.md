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

### Distributed Training Safety Rules

Treat DDP training, validation, checkpointing, and image generation as one synchronized distributed program. Do not assume code is safe just because only rank 0 runs it.

**Rank-0-only inference/sample saving must never build autograd graphs.**

Any validation, sampling, visualization, metric generation, or checkpoint preview code must run under `torch.no_grad()` or `torch.set_grad_enabled(False)`. This is especially important for rank-0-only sample/save blocks, because rank 0 can silently consume much more memory than the other ranks.

Good pattern:

```python
if rank == 0:
    grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        # sample_from_model(...), generator previews, torchvision.save_image(...)
        ...
    finally:
        torch.set_grad_enabled(grad_enabled)
```

Equivalent:

```python
if rank == 0:
    with torch.no_grad():
        ...
```

Do the same for validation loops. Validation is inference, not training.

**Avoid DDP-wrapped modules inside rank-local inference paths.**

When calling generators for validation or rank-0-only sampling, prefer the underlying module:

```python
gen1 = gen_diffusive_1.module if hasattr(gen_diffusive_1, "module") else gen_diffusive_1
```

DDP-wrapped forwards can trigger distributed behavior that is surprising when only one rank is executing that code.

**Only rank 0 should write shared metric/checkpoint files unless filenames are rank-specific.**

Never let all ranks write the same `.npy`, `.csv`, `.mat`, `.pth`, or image path concurrently. Either guard the write with `if rank == 0:` or include rank in the filename.

**Use barriers around epoch-boundary work.**

The safe epoch-boundary pattern is:

```python
# all ranks finished training epoch
if rank == 0:
    with torch.no_grad():
        # sample/save/checkpoint
        ...

dist.barrier()  # all ranks wait before validation

with torch.no_grad():
    # validation on every rank, or rank0-only if intentionally designed
    ...

if rank == 0:
    # write aggregate metrics
    ...

dist.barrier()  # all ranks wait before the next training epoch
```

The post-validation barrier is important. Without it, a faster rank can enter the next epoch's DDP forward/backward while another rank is still validating or writing metrics.

**Do not diagnose NCCL watchdog timeouts in isolation.**

An NCCL `ALLREDUCE` timeout often means one rank already failed earlier. Always inspect the earliest error in the log, not the loudest later error. Search for:

```bash
grep -Ei "OutOfMemory|CUDA out|Traceback|Watchdog|ALLREDUCE|ChildFailedError|killed|exception" <log>
```

If one rank OOMs or crashes, the remaining ranks may later report NCCL timeouts because the collective can no longer complete.

### Known V100/NCCL Incident and Fix

On 2026-05-11, an 8 x V100 real-data run looked like an NCCL deadlock near the beginning of epoch 2. The actual first failure was rank 0 CUDA OOM. The old failing log showed GPU0 using about 31.71 GiB on a 32 GiB V100, with only about 14 MiB free, then the other ranks timed out in NCCL `ALLREDUCE`.

Cause:

- rank 0 performed epoch-end sample/image generation with autograd enabled;
- `sample_from_model(...)` and generator preview calls retained large inference graphs;
- only rank 0 executed this path, so only rank 0's memory climbed;
- after rank 0 OOMed, the other ranks waited in DDP/NCCL synchronization and eventually hit watchdog timeouts;
- older validation code also used DDP-wrapped generator paths, had all ranks writing some metric files, and lacked a post-validation barrier.

Why it may appear fine on larger GPUs:

- the same bug can survive on 48 GB GPUs such as 4090D because there is more memory headroom;
- that does not make the code correct. It only hides the rank-0 memory spike.

Fixes currently expected in `train.py`:

- rank-0 sample/save runs with gradients disabled;
- validation runs with gradients disabled;
- validation/sample paths use non-DDP module objects where appropriate;
- shared metric writes are rank-0-only;
- a post-validation `dist.barrier()` exists before the next epoch;
- `--debug_sync_trace` can print rank-level epoch-boundary progress;
- process group timeout and shutdown are explicit.

The fixed real-data verification completed 10 epochs on 8 x V100:

- launcher: `/NAS_writeable/SynDiff/run_v100_15T_7T_10epoch_fixed.sh`
- log: `/NAS_writeable/SynDiff/checkpoints/syn_V100x8_15T_7T_10epoch_fixed.driver.log`
- metrics: `/NAS_writeable/SynDiff/checkpoints/syn_V100x8_15T_7T_10epoch_fixed/metrics.csv`
- report: `/NAS_writeable/SynDiff/V100_NCCL_DEADLOCK_DEBUG_REPORT.md`

During the fixed run, training memory stayed around 15.5-16.0 GiB per V100, and rank 0 dropped to roughly 6.5 GiB at sample/save boundaries instead of climbing toward 31.7 GiB.

## Checkpoint Structure

Checkpoints (`content.pth`) contain full training state for resuming:
- Generator/discriminator state dicts (both diffusive and non-diffusive)
- Optimizer and scheduler states
- Training metadata (epoch, global_step, args)

Individual generator checkpoints are saved as `{name}_{epoch}.pth` with EMA-swapped weights if `--use_ema` is enabled.

## Sample Data

Sample toy data is available in `SynDiff_sample_data/` containing `T1.mat` and `T2.mat` for testing the pipeline.
