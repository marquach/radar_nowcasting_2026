# data exploration
import matplotlib.pyplot as plt
import datetime as dt
from pilot.helper import parse_custom_timestamp
import numpy as np
import pysteps.visualization as pyvis
from pysteps import io, rcparams
from pilot.get_data import download_mch_hdf5
import os
import torch

date = dt.datetime.strptime("20260331", "%Y%m%d")
BASEURL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-radar-precip"
dates = [date + dt.timedelta(days=x) for x in range(2)]

#[download_mch_hdf5('rzc', BASEURL, date, window = 0) for date in dates]

path = 'pilot/data'
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

# data validation
temp = np.lib.stride_tricks.sliding_window_view(hdf5_files[1], 2)
#check if diff in timestamps is always 5min
np.all(np.diff(temp) == dt.timedelta(seconds = 300)) 

importer_kwargs = rcparams.data_sources['mch']["importer_kwargs"]
importer_kwargs.update({'product': 'rzc'})

len(hdf5_files[0])
io.import_mch_hdf5(hdf5_files[0][1])
R, _, metadata = io.read_timeseries(hdf5_files, importer= io.import_mch_hdf5, **importer_kwargs)

R[:1,:,:].dtype
isinstance(torch.tensor(R[:1,:,:]), torch.Tensor)
import torch.nn.functional as F
temp = np.nan_to_num(R[:1,:,:])
R_tensor = torch.tensor(R[:1,:,:])
R_tensor = R_tensor.unsqueeze(1)
R_tensor.shape
F.interpolate(R_tensor, (64,64))

pyvis.plot_precip_field(R[-1,:,:])
plt.show()
pyvis.animations.animate(R[:52,:,:])

