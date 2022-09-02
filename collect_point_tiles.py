import os
from datetime import datetime
from itertools import product 
import time
import ee

from sattelitetiles import satteliteTiles, extract_sentinel2_pointtile

# Initialization Google Earth Engine
ee.Initialize()

# automatic generation of dateslotes for tiles generation
# years of observation for model 
years = range(2018, datetime.now().year)
# start_month start_day finish_month finish_day
# each line start-finish dates of timeslot:
# spring
# early summer
# late summer
# autumn
dates = [[4, 10, 5,  15],
         [5, 16, 6,  30],
         [7, 1,  8,  10],
         [9, 20, 10, 30]]
dateslots = [[datetime(y, d[0], d[1]), datetime(y, d[2], d[3])] for y, d in product(years, dates)]

# create object <satteliteTiles> for saving tiles and their data into csv
datafolder = os.path.expanduser('~/mybox/eScience_ns/temp0')
os.makedirs(datafolder, exist_ok=True)
data_collector = satteliteTiles(datafolder)

# crs of data must have meter units (not degree). 
# Convert crs before feeding in the script in other case
crs = 'EPSG:32635'
from shapely.geometry import Point
dot = Point(687219.0, 5726132.1)
extract_sentinel2_pointtile(data_collector, point=dot, 
                            dates=dateslots, crs=crs, square=320, 
                            fields={'id': 41})
