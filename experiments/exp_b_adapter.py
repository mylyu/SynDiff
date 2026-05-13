"""Experiment B: Feed SynDiff .mat data into CUT training loop.

Wraps our .mat data (via LoadDataSet) into CUT's {"image": tensor} dict format.
"""
import os, sys
import torch
from torch.utils.data import Dataset

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
from dataset import LoadDataSet


class CUTMatDataAdapter(Dataset):
    """Wraps a .mat file into CUT's {"image": tensor} format.

    Args:
        input_path: Directory containing data_{phase}_{contrast}.mat files
        contrast: Field strength name (e.g., '1.5T', '7T')
        phase: 'train', 'val', or 'test'
    """
    def __init__(self, input_path, contrast, phase="train"):
        mat_path = os.path.join(input_path, f"data_{phase}_{contrast}.mat")
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Mat file not found: {mat_path}")
        data = LoadDataSet(mat_path)   # (N, 1, 256, 256) float32 in [-1, 1]
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"image": self.data[index]}
