""" Function for retrieving square fragments of Sentinel-2 data from
Google Earth Engine for selected time slotes
"""

import ee
from datetime import datetime
import time
from shapely.geometry import Point
from .geeloopsentineltiles import get_aoi, get_geohash, get_sentinel2_tile
from .sattelitetiles import satteliteTiles


def extract_sentinel2_pointtile(data_collector: satteliteTiles,
                                point: Point,
                                dates: list, # of datetime.datetime [start, finish]
                                crs: str,
                                square: int = 320,
                                fields: dict = {}
                                ):
    """
    Collects tile arrays for point with coordinates in <point>
    for dates inside slotes described in <dates>
 
    Parameters
    ----------
    data_collector: satteliteTiles
      object of satteliteTiles class for manipulations with
      files of tile arrays and context information in pandas.DataFrame format
    point: shapely.geometry.Point
      point for extraction Sentinel-2 tiles
    dates: list of datetime.datetime pairs
      intervals of dates for tiles search in;  [[start, finish], ...]
    crs: str
      string representing crs of the point ready to pass in ee.Projection
      for example: f'EPSG:{geoDataFrame.geometry.crs.to_authority()[1]}'
      for correct results units of crs of the point must meters
    square=320
      lenght of tile side (square) in units of crs (m)
    fields : dict {'property': value,...}
      collection of context information properties to include in <data_collector>

    Returns
    -------
    The function updates DataFrame connected with <data_collector> 
    by information about collected tiles and save tile arrays in .npy files
    """

    crs_ee = ee.Projection(crs)

    # defining square area of interest around point
    # for bands with spatial resolution 10m we have to reduse 
    # sides of the square on 10 m for 10*size_in_pixels == size_in_meters
    aoi = get_aoi(point, crs_ee, sz=square-10)
    print(f'The area of interest is {aoi.getInfo()}')
    point_geohash = get_geohash(point, crs)

    # iterates in dateslotes
    for slot in dates :
        # we can't search satelite shoots from the future
        if slot[0] > datetime.now() :
            break
        time.sleep(0.1)
        start = ee.Date(slot[0])
        finish = ee.Date(slot[1])
        # search tiles in aoi inside dateslot
        tiledata = get_sentinel2_tile(aoi, start, finish)
        if not tiledata :
            # if no appropriate product found
            continue
        # update <data_collector> with new data (and save it)
        fields['product_id'] = tiledata['id']
        data_collector.add_record(tiledata['tile_10m'], 10.0, tiledata['date'],
                                  point_geohash,
                                  fields,
                                  '10m02030408')
        data_collector.add_record(tiledata['tile_20m'], 20.0, tiledata['date'],
                                  point_geohash,
                                  fields,
                                  '20m050607081112')
