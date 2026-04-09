import datetime as dt
import bisect, os, re, requests
from pathlib import Path
import torch
from  torch.utils.data import Dataset
import numpy as np

# Add after imports
from typing import Any  # Helps inference


# Lazy load heavy modules (Pylance skips deep analysis)
def _load_pysteps():
    from pysteps import rcparams, io
    return rcparams, io
rcparams, io = _load_pysteps()

def parse_gregorian_day(date):
    gregorian_day = (date.toordinal() - dt.date(date.year,1,1).toordinal())+1
    gregorian_day = str(gregorian_day)

    if len(gregorian_day) == 1:
        gregorian_day = "0"*2 + gregorian_day
    if len(gregorian_day) == 2:
        gregorian_day = "0" + gregorian_day
    return gregorian_day

def parse_utc_time(dt_obj):
    if not isinstance(dt_obj, dt.datetime):
        raise TypeError("Input must be datetime object")
    # only hour & minutes of interest
    hour = str(dt_obj.hour)
    minute = str(dt_obj.minute)

    if len(minute)==1:
        minute = "0" +str(minute)
    if len(hour) == 1:
        hour = "0" + str(hour)
    return hour+minute



# Helper for time period
def is_timepoint(dt_obj):
    if not isinstance(dt_obj, dt.datetime):
        raise TypeError("Input must be datetime object")
    return dt_obj.hour > 0 or dt_obj.minute > 0 or dt_obj.second > 0



# Module-level constant — computed once
UTC_COMBS = [f"{x:02d}{y:02d}" for x in range(24) for y in range(0, 60, 5)]

def download_mch_hdf5(product: str, baseurl: str, date: dt.datetime, window: int, save_dir: str = "pilot/data/"):
    
    if not isinstance(date, dt.datetime):
        raise TypeError("Input must be a datetime object")

    date_str    = date.strftime("%Y%m%d")
    date_url    = date_str + "-ch"
    gregorian_day = parse_gregorian_day(date)
    
    if is_timepoint(date):
        day_time_utc = parse_utc_time(date)
        utc_idx      = max(0, bisect.bisect_right(UTC_COMBS, day_time_utc) - 1)
    else:
        # No time component → download all time steps for the day
        utc_idx = 0
        window  = len(UTC_COMBS)  # 288 steps (24h × 12 per hour)

    urls = [
        "/".join([baseurl, date_url, f"{product}{str(date.year)[-2:]}{gregorian_day}{UTC_COMBS[idx]}vl.001.h5"])
        for idx in range(utc_idx, utc_idx + window) if idx < len(UTC_COMBS)
    ]

    save_path = Path(save_dir) / date_str
    save_path.mkdir(parents=True, exist_ok=True)

    for url in urls:
        filename = url.split("/")[-1]
        response = requests.get(url)

        if response.status_code == 403:
            url      = re.sub(r"([1-9][0-9]*)vl\.", r"\1ul.", url)  # fixed escaped dot
            response = requests.get(url)

        if response.status_code == 200:
            (save_path / filename).write_bytes(response.content)
            print(f"Saved: {filename}")
        else:
            print(f"Download of {filename} failed: {response.status_code}")


data_source = rcparams.data_sources["mch"]
path_fmt = data_source['path_fmt']
fn_pattern = 'rzc%y%j%H%Mvl.001'
fn_ext = 'h5'
importer_name = 'mch_hdf5'
timestep = data_source["timestep"]

importer_kwargs = rcparams.data_sources['mch']["importer_kwargs"]
importer_kwargs.update({'product': 'rzc'})


class rain_precipitation_mch(Dataset):
    def __init__(self, root_dir, date, window_size=7, target_size=1, transform=None):
        self.window_size = window_size
        self.target_size = target_size
        self.transform = transform
        
        fns = io.find_by_date(
            date,
            root_dir,
            path_fmt,
            fn_pattern,
            fn_ext,
            timestep,
            num_next_files=window_size + target_size,
        )
        
        
        importer = io.get_method(importer_name, "importer")
        all_data = io.read_timeseries(fns, importer, **importer_kwargs)
        self.data = all_data[0]
        self.timestamp = all_data[2]['timestamps']
        metadata = all_data[2]
        metadata.pop('timestamps')
        self.metadata = metadata

        if len(self.data) < self.window_size + self.target_size:
            raise ValueError(f"Not enough data: {len(self.data)} < {self.window_size + self.target_size}")
        

    def __len__(self):
        return len(self.data) - self.window_size - self.target_size + 1

    def __getitem__(self, index):
        x = self.data[index : index + self.window_size]
        x = np.nan_to_num(x)
        y = self.data[index + self.window_size : index + self.window_size + self.target_size]
        y = np.nan_to_num(y)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    
    @property
    def timestamp(self):
        return self.timestamp  