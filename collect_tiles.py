import sys
import os
from datetime import datetime
from itertools import product
import time
import ee
# open geodataframe about forest parcels in the PSRER
from sattelitetiles import openvydelpsrer, satteliteTiles
# loop in Sentinel-2 products and saving tiles
from sattelitetiles import extract_sentinel2_inside_geodataframe

# Authentification in Google Earth Engine if needed
# ee.Authenticate()
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
dateslots = [[datetime(y, d[0], d[1]), datetime(y, d[2], d[3])]
             for y, d in product(years, dates)]

# load GeoDataFrame of PSRER
vydela = openvydelpsrer.load_forest_geodatabase()

# columns from geoDataFrame (vydela) which will included into data with tiles
columns = ['NUM_LCH', 'NUM_KV', 'NUM_VD', 'FORESTCODE']

# create object <satteliteTiles> for saving tiles and their data into csv
datafolder = os.path.expanduser('~/mybox/eScience_ns/temp')
data_collector = satteliteTiles(datafolder)


def main(n_rois: int):
    """
    Collecting of Sentinel-2 samples in <n_rois> sites
    """
    for _ in range(n_rois):
        extract_sentinel2_inside_geodataframe(
            data_collector=data_collector,
            areas=vydela,
            dates=dateslots,
            columns=columns
        )
        time.sleep(20)


if __name__ == '__main__':
    # if the script started in Org Babel
    if 'ipykernel' in sys.argv[0]:
        main(1)
    else:
        if len(sys.argv) > 2:
            datafolder = sys.argv[1]
        if len(sys.argv) < 2:
            n_rois = 1
        else:
            try:
                n_rois = int(sys.argv[1])
            except:
                sys.exit("First argument of the script must a number of ROIs")
        main(n_rois)
