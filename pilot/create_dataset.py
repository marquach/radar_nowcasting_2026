
from torch.utils.data import Dataset
import os, torch
from pilot.helper import parse_custom_timestamp
import numpy as np
from pysteps import io
import torch.nn.functional as F
import torch


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

        # now each element is (path, datetime)
        hdf5_files.sort(key=lambda x: x[1])

        paths, times = zip(*hdf5_files)
        hdf5_files = [list(paths), list(times)]
        self.data, _, self._metadata = io.read_timeseries(hdf5_files,
                                                           importer= io.import_mch_hdf5, **{'product': 'rzc', 'unit': 'mm', 'accutime': 5})
        
        if len(self.data) < self.window_size + self.target_size:
            raise ValueError(f"Not enough data: {len(self.data)} < {self.window_size + self.target_size}")
        

    def __len__(self):
        return len(self.data) - self.window_size - self.target_size + 1

    def __getitem__(self, index):    
        x = self.data[index : index + self.window_size, :, :]
        x = np.nan_to_num(x)
        x = torch.tensor(x)
        x = x.unsqueeze(1)
        x = F.interpolate(x, (64,64), mode='bilinear')
        
        y = self.data[index + self.window_size : index + self.window_size + self.target_size, :, :]
        y = np.nan_to_num(y)
        y = torch.tensor(y)
        y = y.unsqueeze(1)
        y = F.interpolate(y, (64,64), mode='bilinear')


        return x, y
    
    @property
    def metadata(self):
        return self._metadata