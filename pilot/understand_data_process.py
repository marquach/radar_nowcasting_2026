import matplotlib.pyplot as plt
import numpy as np
import requests
import itertools
import bisect

from pilot.helper import parse_gregorian_day, parse_utc_time, download_mch_hdf5

from pysteps import rcparams, io
import datetime as dt
from pysteps.utils import conversion, dimension, transformation
from pysteps.visualization import plot_precip_field


date = dt.datetime.strptime("202604051202", "%Y%m%d%H%M")

## https://data.geo.admin.ch/ch.meteoschweiz.ogd-radar-precip/ für REST API später 
# https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-radar-precip api

BASEURL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-radar-precip"
window = 7
date_url = date.__format__("%Y%m%d") + '-ch'
# how to get gregorian day of year
gregorian_day = parse_gregorian_day(date)

utc_combs = [ ["0"+str(x) if len(str(x)) == 1 else str(x), "0"+str(y) if len(str(y)) == 1 else str(y)] for x in range(24) for y in range(0,60,5)]
utc_combs = ["".join(pair) for pair in utc_combs]
# wie bekomme ich utc comb idx

day_time_utc = parse_utc_time(date)
utc_idx = bisect.bisect_right(utc_combs, day_time_utc)-1
product = 'rzc'
test_url = ["/".join([BASEURL, date_url, product + str(date.year)[-2:] + gregorian_day + utc_combs[idx]+"vl.001.h5"]) for idx in range(utc_idx, utc_idx+window)]



test = io.import_mch_hdf5('pilot/data.h5', 'RATE')
R = test[0]
metadata = test[2]
R
R, metadata = conversion.to_raindepth(R, metadata)

# Upscale data to 2 km to limit memory usage
R, metadata = dimension.aggregate_fields_space(R, metadata, 2000)

# Plot the rainfall field
plot_precip_field(R[ :, :], geodata=metadata)
plt.show()

# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
# set the fill value to -15 dBR
R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)

# Set missing values with the fill value
R[~np.isfinite(R)] = -15.0