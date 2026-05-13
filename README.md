# MRIxFields 2026 — CUT + SynDiff

Repository for the MRIxFields2026 challenge (Task 1 & 2). Primary model: **CUT** (contrastive unpaired translation). SynDiff experimental.

Based on the [official MRIxFields2026 baseline](https://github.com/MRIxFields/MRIxFields2026) and the [SynDiff paper](https://ieeexplore.ieee.org/document/10167641).

## Setup

```bash
git clone https://github.com/mylyu/SynDiff
source activate.sh
```

Code lives on NFS at `/NAS_writeable/SynDiff/code/` — servers read from this path directly.

## Servers

| Server | GPUs | Data |
|--------|------|------|
| Local | 6× A40 (48 GB) | All `.mat` + npz |
| 4090D | 2× RTX 4090D (48 GB) | `.mat` |
| V100 | 8× V100 (32 GB) | `.mat` |

## Data

Preprocessed `.mat` files (256×256, [-1,1]) at `/data0/syndiff_data/`:
- `0.1T_to_7T/` — 0.1T ↔ 7T T1W (1/4 subset)
- `1.5T_to_7T/` — 1.5T ↔ 7T T1W (1/4 subset)
- `0.1T_to_1.5T/` — 0.1T ↔ 1.5T T1W (1/4 subset)

npz cache for baseline pipeline at `/data0/syndiff_data/npz_cache_quarter/`.

Raw NIfTI source files at `/data0/MRIxFields2026/TrainingData/release_20260414/` (local only).

## Training

### CUT (primary)

```bash
# Single GPU
python3 code/experiments/run_exp_b.py \
  --input_path /data0/syndiff_data/1.5T_to_7T \
  --output_dir /NAS_writeable/SynDiff/checkpoints \
  --exp CUT_15T_7T --src_field 1.5T --tgt_field 7T \
  --batch_size 8 --num_epoch 100 --lr 2e-4 --device cuda:0
```

### CUT launcher (for remote servers)

```bash
ssh <IP> "nohup bash /NAS_writeable/SynDiff/launch_cut_generic.sh \
  <gpu> <src_field> <tgt_field> <data_dir> <exp_name> <epochs> &>/dev/null &"
```

### SynDiff (experimental)

```bash
python3 code/train.py \
  --image_size 256 --exp syn_test \
  --num_channels 2 --num_channels_dae 64 \
  --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 \
  --batch_size 4 --contrast1 1.5T --contrast2 7T --num_epoch 20 \
  --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 \
  --r1_gamma 1.0 --z_emb_dim 256 --t_emb_dim 256 \
  --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --lambda_l1_loss 0.5 \
  --num_process_per_node 2 \
  --input_path /data0/syndiff_data/1.5T_to_7T \
  --output_path /NAS_writeable/SynDiff/checkpoints
```

## Evaluation

- `code/experiments/eval_exp_b.py` — CUT checkpoint evaluation (nRMSE, SSIM)
- `code/experiments/eval_paired.py` — paired test-set eval for SynDiff
- `code/compute_official_metrics.py` — official MRIxFields metrics from saved predictions

## Results

CUT achieves nRMSE ~0.87 on 1.5T→7T with our `.mat` data (20 epochs). Full 100-epoch training in progress.

SynDiff best nRMSE ~1.13 but degrades over time — parked for now.

## Citation

SynDiff paper:
```
@ARTICLE{ozbey_dalmaz_syndiff_2024,
  author={Özbey, Muzaffer and Dalmaz, Onat and Dar, Salman U. H. and Bedel, Hasan A. and Özturk, Şaban and Güngör, Alper and Çukur, Tolga},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Unsupervised Medical Image Translation With Adversarial Diffusion Models}, 
  year={2023},
  volume={42},
  number={12},
  pages={3524-3539},
  doi={10.1109/TMI.2023.3290149}}
```

Baseline code from [MRIxFields2026](https://github.com/MRIxFields/MRIxFields2026), [CUT](https://github.com/taesungp/contrastive-unpaired-translation), [StyleGAN-2](https://github.com/NVlabs/stylegan2), [DD-GAN](https://github.com/NVlabs/denoising-diffusion-gan).
