#!/bin/bash
# SynDiff environment activation
# Usage: source /NAS_writeable/SynDiff/activate.sh

export SYN_ROOT=/NAS_writeable/SynDiff
export VIRTUAL_ENV=$SYN_ROOT/env
export PATH=$VIRTUAL_ENV/bin:$PATH
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_$(whoami)
export XDG_CONFIG_HOME=/tmp/xdg_config_$(whoami)

echo "SynDiff env: $VIRTUAL_ENV"
echo "Python:  $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA:    $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
echo "GPUs:    $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)"
echo ""
echo "Train:   bash $SYN_ROOT/train.sh [num_gpus] [batch] [epochs]"
