import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import requests
import itertools
import bisect
import os


from pilot.helper import parse_gregorian_day, parse_utc_time, download_mch_hdf5, is_timepoint

from pysteps import rcparams, io
import datetime as dt
from pysteps.utils import conversion, dimension, transformation
from pysteps.visualization import plot_precip_field, animations

date = dt.datetime.strptime("202604051955", "%Y%m%d%H%M")
BASEURL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-radar-precip"
window = 36 # 7*5mim=35 min windows
# otherwise pull whole day

download_mch_hdf5('rzc', BASEURL, date, window= window)

data_source = rcparams.data_sources["mch"]
root_path = 'pilot/data/'
path_fmt = data_source['path_fmt']
fn_pattern = 'rzc%y%j%H%Mvl.001'
fn_ext = 'h5'
importer_name = 'mch_hdf5'
timestep = data_source["timestep"]


date_search = dt.datetime.strptime("202604052015", "%Y%m%d%H%M")
# Find the frame in the archive for the specified date
fns = io.find_by_date(
    date_search, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=1, num_next_files=20)

importer_kwargs = rcparams.data_sources['mch']["importer_kwargs"]
importer_kwargs.update({'product': 'rzc'})
# Read the data from the archive
importer = io.get_method(importer_name, "importer")
R, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)

# data validation
temp = np.lib.stride_tricks.sliding_window_view(metadata['timestamps'], 2)
np.all(np.diff(temp) == dt.timedelta(seconds = 300))

R.shape
R[-1,:,:]
plot_precip_field(R[-1,:,:], geodata=metadata)

animations.animate(R)
