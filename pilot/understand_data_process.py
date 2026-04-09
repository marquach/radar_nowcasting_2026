import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from torchvision import transforms


from pysteps import rcparams, io
import datetime as dt
from pysteps.utils import conversion, dimension, transformation
import pysteps.visualization as pyvis
from pilot.helper import rain_precipitation_mch, download_mch_hdf5



date = dt.datetime.strptime("20260402", "%Y%m%d")
BASEURL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-radar-precip"
dates = [date + dt.timedelta(days=x) for x in range(4)]
dates
window = 36 # 7*5mim=35 min windows
# otherwise pull whole day

#[download_mch_hdf5('rzc', BASEURL, date, window= 0) for date in dates]

#download_mch_hdf5('rzc', BASEURL, date, window= window)

root_path = 'pilot/data/'
date_search = dt.datetime.strptime("202604052015", "%Y%m%d%H%M")

# data validation
#temp = np.lib.stride_tricks.sliding_window_view(metadata['timestamps'], 2)
#np.all(np.diff(temp) == dt.timedelta(seconds = 300))



ds = rain_precipitation_mch(root_path, date = date_search, window_size=4, target_size=1)

print(f"Dataset len: {len(ds)}")
x, y = ds[0]
print(f"x.shape: {x.shape}, y.shape: {y.shape}")  # Erwartet: (4,H,W), (1,H,W)
print(f"Metadata keys: {list(ds.metadata.keys())}")

pyvis.plot_precip_field(x[-1])
plt.show()
pyvis.animations.animate(x)


from pysteps.nowcasts import steps
steps.forecast
nowcast = steps.forecast(ds.data[:8],) 
