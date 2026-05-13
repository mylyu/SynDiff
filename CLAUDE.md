# Claude Code Instructions for MRIxField Challenge Project

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
| 4090D | `ssh jupyter-mylyu@10.102.3.251` | 2× RTX 4090D | `.mat` data |
| V100 | `ssh jupyter-mylyu@10.102.3.250` | 8× V100 (32 GB) | `.mat` data |

**SSH access**: `ssh jupyter-mylyu@<IP>` from local server. All servers share the same user (`jupyter-mylyu`) and read code from NFS.

**Check GPU**: `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`

**Launch training**: `ssh <IP> "nohup bash /NAS_writeable/SynDiff/<launch_script>.sh &>/dev/null &"`

**Copy data between servers**: Relay through local machine — `cat` pipe since direct SCP between remote servers requires host key setup.

Activate env: `source /NAS_writeable/SynDiff/activate.sh`

## Non-Negotiable Experiment Hygiene

Every run must be reproducible from a command, a code state, and a result folder. Do not launch
"quick tests" that overwrite an existing result directory.

- Use a unique experiment name for every run. Include model, task, key change, GPU/server, and date if helpful.
- Use a unique output directory and never `rm -rf` an old run unless the user explicitly asked for cleanup.
- Record the exact command in the run folder before or during launch (`command.txt` or visible in `stdout.log`).
- Record code state when possible: `git -C /NAS_writeable/SynDiff/code status --short` and `git rev-parse HEAD`.
- Change one experimental variable at a time unless the run is explicitly labeled as a combined ablation.
- Do not compare runs unless they used the same split, same evaluator, same preprocessing, and same metric definition.
- Prefer short diagnostic runs before long jobs. A bad 50-epoch run is just a slow bug report.
- Do not launch overlapping jobs on the same GPUs. Check GPU occupancy first.

Recommended launch pattern:

```bash
source /NAS_writeable/SynDiff/activate.sh
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
CUDA_VISIBLE_DEVICES=<ids> nohup bash /NAS_writeable/SynDiff/<launcher>.sh \
  > /NAS_writeable/SynDiff/checkpoints/<exp>/driver.log 2>&1 &
```

## GPU Programming Rules

These rules exist because this project has already hit rank-specific OOM, misleading NCCL
timeouts, stale metrics, and source-biased models.

- Inference, validation, sample saving, and metric computation must run under `torch.no_grad()` or `torch.inference_mode()`.
- Rank-0-only sample/save code is still GPU code. Treat it as carefully as training code.
- Do not use DDP-wrapped modules for local rank-only inference if the underlying `.module` can be used safely.
- Only rank 0 should write shared files such as checkpoints, metric CSVs, `.npy`, `.mat`, and PNG panels.
- Put `dist.barrier()` around validation/sample phases when other ranks must wait before the next epoch.
- Never let one rank enter the next train forward/backward while another rank is still validating or writing files.
- In DDP, call `train_sampler.set_epoch(epoch)` every epoch.
- Detach fake samples during discriminator updates unless generator gradients are intentionally needed.
- Avoid `retain_graph=True` unless there is a clear reason. Retained graphs are a common silent memory leak.
- Do not fix OOM by sprinkling `torch.cuda.empty_cache()`. First find who owns the tensors and graphs.
- Log `torch.cuda.max_memory_allocated()` and `torch.cuda.memory_reserved()` when debugging memory.
- Use AMP deliberately. If a module is numerically fragile in fp16, keep that module in fp32 and document why.
- Always switch modes explicitly: `model.train()` for training, `model.eval()` for evaluation.
- Clamp/scale outputs consistently before metrics. Know whether tensors are in `[-1,1]` or `[0,1]`.
- Use `non_blocking=True` only with pinned-memory loaders; otherwise it is not magic.
- Keep `CUDA_VISIBLE_DEVICES` stable inside a run. Remember that local GPU index 0 means "first visible GPU".

## Distributed Debug Rules

NCCL errors are often downstream symptoms. Find the first real error, not the loudest later one.

- When DDP hangs, inspect all rank logs. The first CUDA OOM, Python traceback, or data-loader error is usually the cause.
- Enable `NCCL_ASYNC_ERROR_HANDLING=1` and a finite process-group timeout for long multi-GPU debugging.
- For suspected synchronization bugs, add rank-level trace logs at epoch start, train end, validation start/end, checkpoint start/end, and barrier entry/exit.
- If only rank 0 does extra work, compare rank 0 memory against other ranks at that point.
- Reduce the problem in this order: single batch on one GPU, one epoch on one GPU, two GPUs, then full GPU count.
- Do not trust a distributed run that only survived epoch 1 if the historical failure happens at epoch 2.
- If a worker dies, make the parent process surface the nonzero child exit code. Silent child death is not acceptable.
- Do not reuse a TCP port across concurrent distributed jobs.
- Do not run multiple V100 experiments on GPUs `0,1` by accident. Explicitly set disjoint `CUDA_VISIBLE_DEVICES`.

## Monitoring Rules

Monitoring is part of the experiment, not an afterthought.

Before launch:

```bash
nvidia-smi
ps -fu "$USER" | grep -E "train.py|run_exp|torchrun|python" | grep -v grep
df -h /NAS_writeable /data0
```

During launch:

```bash
tail -f /NAS_writeable/SynDiff/checkpoints/<exp>/stdout.log
tail -f /NAS_writeable/SynDiff/checkpoints/<exp>/stderr.log
nvidia-smi dmon -s pucm
```

Watch for:

- GPU memory climbing every epoch, especially only on rank 0.
- GPU utilization at 0% while processes still hold memory.
- DDP ranks printing different epoch/iteration numbers.
- Repeated D loss around `2.772` in SynDiff, which indicates near-random summed softplus discriminator behavior.
- CUT/CycleGAN loss CSVs being mistaken for official nRMSE/SSIM.
- Validation metrics improving while qualitative panels look worse. That usually means the metric/eval pairing is wrong.
- Any `Traceback`, `CUDA out of memory`, `NCCL watchdog`, `DataLoader worker`, or `IndexError` in logs.

If a job appears stuck, do not immediately kill it. First capture:

```bash
nvidia-smi
ps -fu "$USER" | grep -E "train.py|run_exp|python" | grep -v grep
tail -n 80 /NAS_writeable/SynDiff/checkpoints/<exp>/stdout.log
tail -n 80 /NAS_writeable/SynDiff/checkpoints/<exp>/stderr.log
```

## Verification Gates

A change is not "done" until it passes the right gate for the risk level.

For code edits:

- `python -m py_compile` on edited Python files.
- A smoke command that imports the edited path.
- No accidental edits to unrelated files.
- No new long-running process left behind unless the user asked for it.

For training-loop edits:

- One mini/smoke run reaches validation and writes metrics.
- A multi-GPU smoke run reaches epoch 2 if the bug could be distributed.
- For V100/DDP safety, run long enough to pass the historical failure point.
- Confirm all child processes exit cleanly.

For model-quality claims:

- Use paired evaluation only for nRMSE/SSIM claims.
- Save fixed qualitative panels: source, target, prediction, absolute error.
- Evaluate the same fixed case indices across epochs and experiments.
- Report best checkpoint by official paired metric, not by final epoch.
- Compare prediction-to-source versus prediction-to-target to detect identity/source bias.
- Do not claim CUT, CycleGAN, or SynDiff is better from training losses alone.

For challenge-readiness:

- Run official metrics from saved predictions with `code/compute_official_metrics.py`.
- Verify output file names, shapes, orientation, intensity range, and affine/header handling.
- Keep the exact inference command and checkpoint path in the result folder.
- Inspect at least a few generated NIfTI volumes visually, not just PNG slices.

## Debugging Discipline

Use a tight loop: hypothesis → smallest test → evidence → next decision.

- Write down the suspected failure mode before launching a run.
- Prefer instrumentation over guessing: log logits, grad norms, memory, case IDs, and metric inputs.
- When results surprise you, first verify data pairing, normalization, and evaluator consistency.
- When changing losses, log every loss component separately. Total loss alone hides failures.
- When changing data loaders, log dataset lengths, sample IDs, and whether train/val/test are paired or unpaired.
- Keep failed runs if they contain useful evidence. Move junk to a trash-bin folder during cleanup instead of deleting it.
- If a result contradicts previous evidence, do not bury it. Add a note explaining what changed.

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
