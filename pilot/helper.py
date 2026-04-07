import datetime as dt

def download_mch_hdf5(url: str):
        filename = url.split('/')[-1]
        response = requests.get(url)

        if response.status_code == 200:
            with open('pilot/'+filename, "wb") as f:
                f.write(response.content)
            print("HDF5 file saved as data.h5")
        else:
            print(f"Download failed: {response.status_code}")


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
