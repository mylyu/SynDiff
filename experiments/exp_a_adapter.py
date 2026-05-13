"""Experiment A: Feed baseline npz data into SynDiff training loop.

Wraps two CachedUnpairedDataset instances into SynDiff's (x1, x2) tuple format,
matching the interface of UnpairedSliceDataset.
"""
import os, sys
import torch
from torch.utils.data import Dataset

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from cached_dataset import CachedUnpairedDataset


class SynDiffNPZAdapter(Dataset):
    """Bridge: baseline npz -> SynDiff (x1, x2) tuple pairs.

    Matches UnpairedSliceDataset interface:
        __len__ = max(len(src), len(tgt))
        __getitem__ returns (src_tensor, tgt_tensor)

    Both tensors are (1, H, W) float32 in [-1, 1].
    """

    def __init__(self, preprocessed_dir, split, modality, src_field, tgt_field,
                 crop_size=(256, 256)):
        self.src = CachedUnpairedDataset(
            preprocessed_dir=preprocessed_dir, split=split,
            modality=modality, field_strength=src_field,
            crop_size=crop_size,
        )
        self.tgt = CachedUnpairedDataset(
            preprocessed_dir=preprocessed_dir, split=split,
            modality=modality, field_strength=tgt_field,
            crop_size=crop_size,
        )
        self.length = max(len(self.src), len(self.tgt))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        src_item = self.src[index % len(self.src)]
        tgt_item = self.tgt[(index * 9973) % len(self.tgt)]
        return src_item["image"], tgt_item["image"]
