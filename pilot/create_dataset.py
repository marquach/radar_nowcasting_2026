from torch.utils.data import Dataset
import os, torch
from pilot.helper import parse_custom_timestamp
import numpy as np
from pysteps import io
from pysteps.utils import transformation
import torch.nn.functional as F


class rain_precipitation_mch(Dataset):
    def __init__(self, path, window_size=7, target_size=1, transform=None):
        self.window_size = window_size
        self.target_size = target_size
        self.transform = transform
        
        hdf5_files = []

        for dirpath, dirnames, filenames in os.walk(path):
            for fname in filenames:
                if fname.endswith('.h5'):
                    full_path = os.path.join(dirpath, fname)
                    ts = parse_custom_timestamp(fname[3:12])
                    hdf5_files.append((full_path, ts))

        hdf5_files.sort(key=lambda x: x[1])

        paths, times = zip(*hdf5_files)
        hdf5_files = [list(paths), list(times)]

        self.data, _, self._metadata = io.read_timeseries(hdf5_files,
                                                           importer=io.import_mch_hdf5, **{'product': 'rzc', 'unit': 'mm', 'accutime': 5})
        self.data = np.nan_to_num(self.data)
        
        # dBR transform + min-max normalization
        self.data, self._dbr_metadata = transformation.dB_transform(self.data, self._metadata)
        self.data = np.nan_to_num(self.data)  # dB_transform sets norain to NaN
        self.data_min = self.data.min()
        self.data_max = self.data.max()
        self.data = (self.data - self.data_min) / (self.data_max - self.data_min)

        self.valid_indices = []
        for i in range(len(self.data) - window_size - target_size + 1):
            target = self.data[i + window_size : i + window_size + target_size]
            if np.mean(target > 0) > 0.01:
                self.valid_indices.append(i)
        
        if len(self.data) < self.window_size + self.target_size:
            raise ValueError(f"Not enough data: {len(self.data)} < {self.window_size + self.target_size}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):    
        i = self.valid_indices[index]
        x = self.data[i : i + self.window_size, :, :]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x = F.interpolate(x, (64, 64), mode='bilinear')
        x = x.squeeze(0)
        
        y = self.data[i + self.window_size : i + self.window_size + self.target_size, :, :]
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        y = F.interpolate(y, (64, 64), mode='bilinear')
        y = y.squeeze(0)

        return x, y

    def inverse_transform(self, tensor):
        """Convert model output back to mm/h."""
        arr = tensor.detach().cpu().numpy()
        arr = arr * (self.data_max - self.data_min) + self.data_min  # undo min-max
        arr, _ = transformation.dB_transform(arr, inverse=True)      # undo dBR
        return arr
    
    @property
    def metadata(self):
        return self._metadata