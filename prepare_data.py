#!/usr/bin/env python3
"""Prepare NIfTI data for SynDiff training.

Converts 3D NIfTI volumes to 2D axial slices, center-crops to 256x256,
filters empty slices, splits train/val from retrospective (unpaired),
and creates paired test set from prospective data.

Output: HDF5 .mat files with variable 'data_fs', shape (N, 256, 256), values in [0,1].
"""

import argparse
import os
import sys
import numpy as np
import h5py
import nibabel as nib


def extract_slices(nifti_path, crop_size=256):
    """Load 3D NIfTI and extract axial slices, center-cropped to crop_size.

    Returns (D, crop_size, crop_size) float32 array.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)  # (H, W, D)
    H, W, D = data.shape

    x_start = (H - crop_size) // 2
    y_start = (W - crop_size) // 2

    slices = []
    for z in range(D):
        sl = data[x_start:x_start + crop_size, y_start:y_start + crop_size, z]
        slices.append(sl)

    return np.stack(slices, axis=0)  # (D, crop_size, crop_size)


def filter_slices(slices, threshold=0.01):
    """Keep slices whose mean is above threshold. Returns filtered array and kept indices."""
    means = slices.mean(axis=(1, 2))
    keep = means > threshold
    return slices[keep], keep


def process_retrospective_field(data_dir, contrast, field_strength, crop_size, threshold):
    """Load all subjects for one field strength from retrospective data, extract and filter slices."""
    fs_dir = os.path.join(data_dir, 'Training_retrospective', contrast, field_strength)
    files = sorted([f for f in os.listdir(fs_dir) if f.endswith('.nii.gz')])
    all_slices = []
    print(f"  Processing {len(files)} subjects from {fs_dir} ...")
    for i, fname in enumerate(files):
        path = os.path.join(fs_dir, fname)
        slices = extract_slices(path, crop_size)          # (D, 256, 256)
        slices, _ = filter_slices(slices, threshold)
        all_slices.append(slices)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(files)} done, {sum(s.shape[0] for s in all_slices)} slices so far")
    result = np.concatenate(all_slices, axis=0)
    print(f"  Total: {result.shape[0]} slices from {len(files)} subjects")
    return result


def process_prospective_paired(data_dir, contrast, field_a, field_b, crop_size, threshold):
    """Load prospective subjects and produce paired test slices.

    Only keeps slices where BOTH field strengths pass the threshold filter.
    """
    pro_dir = os.path.join(data_dir, 'Training_prospective', contrast)
    dir_a = os.path.join(pro_dir, field_a)
    dir_b = os.path.join(pro_dir, field_b)

    files_a = sorted([f for f in os.listdir(dir_a) if f.endswith('.nii.gz')])
    files_b = sorted([f for f in os.listdir(dir_b) if f.endswith('.nii.gz')])

    # Match by subject ID (embedded in filename)
    ids_a = [_extract_subject_id(f) for f in files_a]
    ids_b = [_extract_subject_id(f) for f in files_b]
    common_ids = sorted(set(ids_a) & set(ids_b))
    print(f"  Common prospective subjects: {common_ids}")

    slices_a_all, slices_b_all = [], []
    for sid in common_ids:
        fname_a = next(f for f in files_a if _extract_subject_id(f) == sid)
        fname_b = next(f for f in files_b if _extract_subject_id(f) == sid)
        path_a = os.path.join(dir_a, fname_a)
        path_b = os.path.join(dir_b, fname_b)

        vol_a = extract_slices(path_a, crop_size)
        vol_b = extract_slices(path_b, crop_size)

        # Intersection filter: keep slice if BOTH pass
        mask_a = vol_a.mean(axis=(1, 2)) > threshold
        mask_b = vol_b.mean(axis=(1, 2)) > threshold
        keep = mask_a & mask_b

        print(f"  Subject {sid}: {keep.sum()}/{vol_a.shape[0]} paired slices kept")
        slices_a_all.append(vol_a[keep])
        slices_b_all.append(vol_b[keep])

    return np.concatenate(slices_a_all, axis=0), np.concatenate(slices_b_all, axis=0)


def _extract_subject_id(filename):
    """Extract 4-digit subject ID from filename like 'R_T1W_0.1T_0001.nii.gz' or 'P_T1W_0.1T_0006.nii.gz'."""
    return filename.split('_')[-1].replace('.nii.gz', '')


def save_mat(output_path, array):
    """Save numpy array as HDF5 .mat file with 'data_fs' variable."""
    print(f"  Saving {array.shape} to {output_path} ...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data_fs', data=array)


def main():
    parser = argparse.ArgumentParser(description='Prepare NIfTI data for SynDiff')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to MRIxFields2026 release directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output .mat files')
    parser.add_argument('--contrast', type=str, default='T1W',
                        help='MRI contrast (default: T1W)')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Center-crop size (default: 256)')
    parser.add_argument('--empty_threshold', type=float, default=0.01,
                        help='Mean threshold for filtering empty slices (default: 0.01)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling (default: 42)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    field_a = '0.1T'
    field_b = '1.5T'

    # ---- Retrospective (unpaired) ----
    print("\n=== Retrospective: 0.1T ===")
    data_01T = process_retrospective_field(
        args.data_root, args.contrast, field_a,
        args.crop_size, args.empty_threshold)

    print("\n=== Retrospective: 1.5T ===")
    data_15T = process_retrospective_field(
        args.data_root, args.contrast, field_b,
        args.crop_size, args.empty_threshold)

    # Shuffle independently (unpaired)
    print("\n=== Shuffling and splitting (unpaired) ===")
    rng.shuffle(data_01T)
    rng.shuffle(data_15T)

    n_val_01T = int(data_01T.shape[0] * args.val_ratio)
    n_val_15T = int(data_15T.shape[0] * args.val_ratio)

    print(f"  0.1T: {data_01T.shape[0]} total → {data_01T.shape[0] - n_val_01T} train / {n_val_01T} val")
    print(f"  1.5T: {data_15T.shape[0]} total → {data_15T.shape[0] - n_val_15T} train / {n_val_15T} val")

    train_a = data_01T[n_val_01T:]
    train_b = data_15T[n_val_15T:]
    val_a = data_01T[:n_val_01T]
    val_b = data_15T[:n_val_15T]

    # Truncate to equal sizes (TensorDataset requires matching first dims)
    if train_a.shape[0] != train_b.shape[0]:
        min_n = min(train_a.shape[0], train_b.shape[0])
        print(f"  Truncating train sets to min size: {min_n}")
        train_a, train_b = train_a[:min_n], train_b[:min_n]
    if val_a.shape[0] != val_b.shape[0]:
        min_n = min(val_a.shape[0], val_b.shape[0])
        print(f"  Truncating val sets to min size: {min_n}")
        val_a, val_b = val_a[:min_n], val_b[:min_n]

    save_mat(os.path.join(args.output_dir, f'data_train_{field_a}.mat'), train_a)
    save_mat(os.path.join(args.output_dir, f'data_val_{field_a}.mat'), val_a)
    save_mat(os.path.join(args.output_dir, f'data_train_{field_b}.mat'), train_b)
    save_mat(os.path.join(args.output_dir, f'data_val_{field_b}.mat'), val_b)

    # ---- Prospective (paired) ----
    print("\n=== Prospective: Paired test set ===")
    test_a, test_b = process_prospective_paired(
        args.data_root, args.contrast, field_a, field_b,
        args.crop_size, args.empty_threshold)

    print(f"  Paired test slices: {test_a.shape[0]}")
    save_mat(os.path.join(args.output_dir, f'data_test_{field_a}.mat'), test_a)
    save_mat(os.path.join(args.output_dir, f'data_test_{field_b}.mat'), test_b)

    print("\n=== Done ===")
    for fname in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, fname)
        sz_mb = os.path.getsize(fpath) / (1024 * 1024)
        with h5py.File(fpath, 'r') as f:
            shp = f['data_fs'].shape
        print(f"  {fname}: shape={shp}, size={sz_mb:.1f} MB")


if __name__ == '__main__':
    main()
