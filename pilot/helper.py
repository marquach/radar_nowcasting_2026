import datetime as dt

def parse_custom_timestamp(s: str) -> dt:
    # s format: 'YYDDDHHMM'
    year = 2000 + int(s[0:2])      # '26' -> 2026
    day_of_year = int(s[2:5])      # '094' -> 94th day of year
    hour = int(s[5:7])             # '15' -> 15
    minute = int(s[7:9])           # '05' -> 5

    # Start from Jan 1 and add (day_of_year - 1) days
    base_date = dt.datetime(year, 1, 1) + dt.timedelta(days=day_of_year - 1)

    # Set hour and minute
    return base_date.replace(hour=hour, minute=minute)
