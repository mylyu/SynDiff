# Claude Code Instructions for SynDiff Project

## Code Location

All code lives at `/NAS_writeable/SynDiff/code/` (NFS), symlinked as `code/` from this directory.
This is the git working tree → `github.com/mylyu/SynDiff`.

To commit: `git -C /NAS_writeable/SynDiff/code add ... && git -C /NAS_writeable/SynDiff/code commit ... && git -C /NAS_writeable/SynDiff/code push`

Servers read directly from NFS: `/NAS_writeable/SynDiff/code/train.py`

## Current Focus: CUT (not SynDiff)

SynDiff's diffusive module degrades over time regardless of hyperparameters (lambda_l1,
pre-trained non-diff, removed R1, spectral norm). CUT achieves nRMSE 0.87 on 1.5T→7T
with our .mat data and is the primary model now.

**CUT model files** (in `code/`): `cut_model.py`, `networks.py`, `patchnce.py`, `adversarial.py`, `unpaired_loader.py`

**CUT training**: `code/experiments/run_exp_b.py` — uses `exp_b_adapter.py` to wrap .mat data

**CUT launcher**: `/NAS_writeable/SynDiff/launch_cut_generic.sh <gpu> <src> <tgt> <data_dir> <exp> <epochs>`

## Servers

| Server | IP | GPUs | Data |
|--------|----|------|------|
| Local | — | 6× A40 (48 GB) | All .mat data + npz |
| 4090D | 10.102.3.251 | 2× RTX 4090D | `.mat` data |
| V100 | 10.102.3.250 | 8× V100 (32 GB) | `.mat` data |

Activate env: `source /NAS_writeable/SynDiff/activate.sh`

## Data

**1/4 subset file list**: `retrospective_front25_filelist.txt`
**Raw NIfTI**: `/data0/MRIxFields2026/TrainingData/release_20260414/Training_retrospective/` (local only)

**.mat data** (1/4 subset, 256×256, [-1,1]):
- Local: `/data0/syndiff_data/0.1T_to_7T/`, `1.5T_to_7T/`, `0.1T_to_1.5T/`
- 4090D: same paths
- V100: same paths

**npz cache** (for baseline pipeline):
- Full: `/data0/syndiff_data/npz_cache_retro_T1W/` (local + V100)
- Quarter: `/data0/syndiff_data/npz_cache_quarter/` (local + V100)

## Current Experiments (all CUT)

| Server | GPUs | Task | Epochs | Status |
|--------|------|------|--------|--------|
| Local 0-2 | 3 | 1.5T/0.1T/0.1T→1.5T | 100 | ~37/100 |
| V100 0-7 | 8 | Various | 30-100 | Running |
| 4090D 0-1 | 2 | 1.5T→7T | 100 | Running |

## PDF Improvement Directions

From `MRIxField Task1-2 Model Improvement Suggestions.pdf` (local):

1. **z_map** (slice position encoding) — add normalized slice index as extra channel. CUT needs `input_nc=2`. (Attempted, needs debugging)
2. **2.5D input** — use [z-1, z, z+1] as multi-channel input
3. **Structure losses** — SSIM + edge_loss (Sobel) in fine-tuning: `L = L1 + 0.2*SSIM + 0.1*LPIPS + 0.05*edge_loss`
4. **Multi-condition model** — embed field + modality + slice position
5. **Checkpoint ensemble** — average best 3-5 checkpoints, CUT + CycleGAN fusion

## Key SynDiff Findings (archived)

- D loss stuck at ~2.772 = dead discriminator
- Non-diffusive generators completely broken (SSIM 0.002) inside SynDiff training
- Standalone CycleGAN proves ResnetGenerator CAN learn (nRMSE 1.0, SSIM 0.26)
- All lambda_l1 values degrade (0.5, 5.0, 10.0, 20.0)
- R1 gradient penalty removal doesn't help
- beta2=0.999 made things worse
- Paired eval revealed Dir1 (7T→0.1T) has source bias

## Evaluation

- `code/experiments/eval_paired.py` — paired test-set eval for SynDiff (diffusive)
- `code/experiments/eval_nondiff.py` — non-diffusive generator eval
- `code/experiments/eval_exp_b.py` — CUT checkpoint evaluation (nRMSE, SSIM)
- `code/compute_official_metrics.py` — official MRIxFields metrics from saved preds
