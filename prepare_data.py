#!/usr/bin/env python3
"""Prepare NIfTI data for SynDiff training.

Converts 3D NIfTI volumes to 2D axial slices, center-crops to 256x256,
filters empty slices, splits train/val from retrospective (unpaired),
and creates paired test set from prospective data.

Output: HDF5 .mat files with variable 'data_fs', shape (N, 256, 256), values in [0,1].
"""

import argparse
import os
import json
import numpy as np
import h5py
import nibabel as nib


def extract_slices(nifti_path, crop_size=256, slice_start=None, slice_end=None):
    """Load 3D NIfTI and extract axial slices, center-cropped to crop_size.

    Returns (N, crop_size, crop_size) float32 array.
    """
    img = nib.load(nifti_path)
    img = nib.as_closest_canonical(img)       # reorient to RAS+
    data = img.get_fdata().astype(np.float32)  # (H, W, D)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, 1.0)
    H, W, D = data.shape

    # Exclude background-heavy slices at top/bottom (like baseline: middle ~60%)
    if slice_start is None:
        slice_start = 0
    if slice_end is None:
        slice_end = D
    z_range = range(max(0, slice_start), min(D, slice_end))

    if H < crop_size or W < crop_size:
        raise ValueError(
            f"{nifti_path} has in-plane shape {(H, W)}, smaller than crop_size={crop_size}"
        )

    x_start = (H - crop_size) // 2
    y_start = (W - crop_size) // 2

    crop = data[x_start:x_start + crop_size, y_start:y_start + crop_size, :]
    slices = crop[:, :, z_range].transpose(2, 0, 1)  # (N, crop_size, crop_size)

    return np.ascontiguousarray(slices)


def filter_slices(slices, threshold=0.01):
    """Keep slices whose std is above threshold (baseline approach: std > mean for anatomy detection)."""
    stds = slices.std(axis=(1, 2))
    keep = stds > threshold
    return slices[keep], keep


def process_retrospective_field(data_dir, contrast, field_strength, crop_size, threshold,
                                file_list=None, slice_start=None, slice_end=None):
    """Load all subjects for one field strength from retrospective data, extract and filter slices.

    If file_list is provided, only files whose relative path (e.g.
    Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz) is in the set will be used.
    slice_start/slice_end: optional spatial range to exclude background-heavy top/bottom slices.
    """
    fs_dir = os.path.join(data_dir, 'Training_retrospective', contrast, field_strength)
    files = sorted([f for f in os.listdir(fs_dir) if f.endswith('.nii.gz')])
    if file_list is not None:
        allowed = set(file_list)
        files = [f for f in files if f'Training_retrospective/{contrast}/{field_strength}/{f}' in allowed]
    all_slices = []
    print(f"  Processing {len(files)} subjects from {fs_dir} ...")
    for i, fname in enumerate(files):
        path = os.path.join(fs_dir, fname)
        slices = extract_slices(path, crop_size, slice_start, slice_end)  # (N, 256, 256)
        slices, _ = filter_slices(slices, threshold)
        all_slices.append(slices)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(files)} done, {sum(s.shape[0] for s in all_slices)} slices so far")
    result = np.concatenate(all_slices, axis=0)
    print(f"  Total: {result.shape[0]} slices from {len(files)} subjects")
    return result


def process_prospective_paired(data_dir, contrast, field_a, field_b, crop_size, threshold,
                                slice_start=None, slice_end=None):
    """Load prospective subjects and produce paired test slices.

    Only keeps slices where BOTH field strengths pass the std threshold filter.
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

        vol_a = extract_slices(path_a, crop_size, slice_start, slice_end)
        vol_b = extract_slices(path_b, crop_size, slice_start, slice_end)

        # Intersection filter: keep slice if BOTH pass std threshold
        mask_a = vol_a.std(axis=(1, 2)) > threshold
        mask_b = vol_b.std(axis=(1, 2)) > threshold
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
    array = np.asarray(array, dtype=np.float32)
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
    parser.add_argument('--field_a', type=str, default='0.1T',
                        help='Source field strength (default: 0.1T)')
    parser.add_argument('--field_b', type=str, default='1.5T',
                        help='Target field strength (default: 1.5T)')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Center-crop size (default: 256)')
    parser.add_argument('--empty_threshold', type=float, default=0.01,
                        help='Std threshold for filtering empty slices (default: 0.01)')
    parser.add_argument('--slice_start', type=int, default=None,
                        help='First axial slice index to include (excludes background top)')
    parser.add_argument('--slice_end', type=int, default=None,
                        help='Last axial slice index to include (excludes background bottom)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling (default: 42)')
    parser.add_argument('--file_list', type=str, default=None,
                        help='Optional path to a .txt listing specific NIfTI files to use')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    field_a = args.field_a
    field_b = args.field_b

    # ---- File list (optional subset) ----
    file_list = None
    if args.file_list:
        with open(args.file_list) as f:
            file_list = set(line.strip() for line in f if line.strip())
        print(f"\n=== Using file list: {len(file_list)} allowed files ===")

    # ---- Retrospective (unpaired) ----
    print(f"\n=== Retrospective: {field_a} ===")
    data_01T = process_retrospective_field(
        args.data_root, args.contrast, field_a,
        args.crop_size, args.empty_threshold, file_list,
        args.slice_start, args.slice_end)

    print(f"\n=== Retrospective: {field_b} ===")
    data_15T = process_retrospective_field(
        args.data_root, args.contrast, field_b,
        args.crop_size, args.empty_threshold, file_list,
        args.slice_start, args.slice_end)

    # Shuffle independently (unpaired)
    print("\n=== Shuffling and splitting (unpaired) ===")
    rng.shuffle(data_01T)
    rng.shuffle(data_15T)

    n_val_01T = int(data_01T.shape[0] * args.val_ratio)
    n_val_15T = int(data_15T.shape[0] * args.val_ratio)

    print(f"  {field_a}: {data_01T.shape[0]} total -> {data_01T.shape[0] - n_val_01T} train / {n_val_01T} val")
    print(f"  {field_b}: {data_15T.shape[0]} total -> {data_15T.shape[0] - n_val_15T} train / {n_val_15T} val")

    save_mat(os.path.join(args.output_dir, f'data_train_{field_a}.mat'),
             data_01T[n_val_01T:])
    save_mat(os.path.join(args.output_dir, f'data_val_{field_a}.mat'),
             data_01T[:n_val_01T])
    save_mat(os.path.join(args.output_dir, f'data_train_{field_b}.mat'),
             data_15T[n_val_15T:])
    save_mat(os.path.join(args.output_dir, f'data_val_{field_b}.mat'),
             data_15T[:n_val_15T])

    # ---- Prospective (paired) ----
    print("\n=== Prospective: Paired test set ===")
    test_a, test_b = process_prospective_paired(
        args.data_root, args.contrast, field_a, field_b,
        args.crop_size, args.empty_threshold,
        args.slice_start, args.slice_end)

    print(f"  Paired test slices: {test_a.shape[0]}")
    save_mat(os.path.join(args.output_dir, f'data_test_{field_a}.mat'), test_a)
    save_mat(os.path.join(args.output_dir, f'data_test_{field_b}.mat'), test_b)

    metadata = {
        'data_root': args.data_root,
        'contrast': args.contrast,
        'field_a': field_a,
        'field_b': field_b,
        'crop_size': args.crop_size,
        'empty_threshold': args.empty_threshold,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'counts': {
            f'train_{field_a}': int(data_01T.shape[0] - n_val_01T),
            f'val_{field_a}': int(n_val_01T),
            f'train_{field_b}': int(data_15T.shape[0] - n_val_15T),
            f'val_{field_b}': int(n_val_15T),
            'test_paired': int(test_a.shape[0]),
        },
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Done ===")
    for fname in sorted(os.listdir(args.output_dir)):
        if not fname.endswith('.mat'):
            continue
        fpath = os.path.join(args.output_dir, fname)
        sz_mb = os.path.getsize(fpath) / (1024 * 1024)
        with h5py.File(fpath, 'r') as f:
            shp = f['data_fs'].shape
        print(f"  {fname}: shape={shp}, size={sz_mb:.1f} MB")


if __name__ == '__main__':
    main()
