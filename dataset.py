import torch
import torch.utils.data
import numpy as np
import h5py


class UnpairedSliceDataset(torch.utils.data.Dataset):
    """Pair independently shuffled domains without requiring equal lengths."""

    def __init__(self, data_a, data_b):
        self.data_a = torch.from_numpy(data_a)
        self.data_b = torch.from_numpy(data_b)
        self.length = max(len(self.data_a), len(self.data_b))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # The source arrays are shuffled independently during preparation.
        # A prime stride avoids locking the shorter domain to the same cycle.
        index_a = index % len(self.data_a)
        index_b = (index * 9973) % len(self.data_b)
        return self.data_a[index_a], self.data_b[index_b]



def CreateDatasetSynthesis(phase, input_path, contrast1='T1', contrast2='T2'):
    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
    data_fs_s1 = LoadDataSet(target_file)

    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
    data_fs_s2 = LoadDataSet(target_file)

    if phase in ('train', 'val'):
        return UnpairedSliceDataset(data_fs_s1, data_fs_s2)
    return torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1), torch.from_numpy(data_fs_s2))


def LoadDataSet(load_dir, variable='data_fs', padding=True, Norm=True):
    with h5py.File(load_dir, 'r') as f:
        raw = np.array(f[variable])

    if raw.ndim == 3:
        data = np.expand_dims(np.transpose(raw, (0, 2, 1)), axis=1)
    else:
        data = np.transpose(raw, (1, 0, 3, 2))
    data = data.astype(np.float32)

    if padding:
        pad_x = max((256 - data.shape[2]) // 2, 0)
        pad_y = max((256 - data.shape[3]) // 2, 0)
        if pad_x or pad_y:
            print('padding in x-y with:' + str(pad_x) + '-' + str(pad_y))
            data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

    if Norm:
        data = (data - 0.5) / 0.5
    return data
