"""CUT adapter with slice position encoding (z_map).

Adds a second channel containing the normalized slice position,
as suggested in the MRIxFields improvement guide.
"""
import os, sys
import torch
from torch.utils.data import Dataset

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
from dataset import LoadDataSet


class CUTMatDataAdapterZMap(Dataset):
    """Wraps .mat data + z_map into CUT's {"image": tensor} format.

    Output tensor shape: (2, H, W) where channel 0 = image, channel 1 = z_map.
    z_map is a constant tensor with the normalized slice position.
    """

    def __init__(self, input_path, contrast, phase="train"):
        mat_path = os.path.join(input_path, f"data_{phase}_{contrast}.mat")
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Mat file not found: {mat_path}")
        data = LoadDataSet(mat_path)   # (N, 1, 256, 256) float32 in [-1, 1]
        self.data = torch.from_numpy(data)
        N = len(self.data)
        # Create z_map: normalized position [0, 1]
        self.z_map = torch.linspace(0, 1, N).view(N, 1, 1, 1).expand(N, 1, 256, 256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"image": torch.cat([self.data[index], self.z_map[index]], dim=0)}
