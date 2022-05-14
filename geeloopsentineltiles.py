import sys
import os
import ee
from datetime import datetime
import time
import numpy as np
import geopandas as gpd

# adding path for start the script from Org Babel or Jupyter Notebook
project_folder = os.path.expanduser('~/mybox/eScience_ns/dzz_pgrez')
sys.path.append(project_folder)

from sattelitetiles import satteliteTiles
from extractsdotcompartment import extract_point_in_compartment

from shapely.ops import transform
from shapely.geometry import Point
import geohash
import pyproj

def get_geohash(point: Point,
                crs: str,
                precission=12) -> str:
  """
  Generates geohash of a point with any crs

  Parameters
  ----------
  point : shapely.geometry.Point
    coordinates (x, y) of a point
  crs : any format supported by pyproj
    crs of a point coordinates
  precission : int, default 12
    lenght of geohash string

  Returns
  -------
  str
    geohash of the point  
    https://en.wikipedia.org/wiki/Geohash
  """

  reproject = pyproj.Transformer.from_proj(
                pyproj.Proj(crs),      # sourse crs
                pyproj.Proj("EPSG:4326"),     # destination crs
                always_xy=True)               # x, y (Lat, Lon) order of coordinates
  point2 = transform(reproject.transform, point)  # apply reprojection
  geohashstr = geohash.encode(point2.y, point2.x, precission)
  return geohashstr

def get_aoi(center: Point, 
            crs: ee.Projection, 
            sz=320) -> ee.Geometry.Polygon :
  """
  Creates rectangle with defined size around point.
  Sides of the rectangle are parallel with a geographic grid

  Parameters
  ----------
  center : shapely.geometry.Point
    coordinates of central point
  crs : ee.Projection
    geographic coordinate system for calculations. Size of rectangle defined by units of the crs
  sz : float
    size of the rectangle side in a units of crs

  Returns
  -------
  ee.Geometry.Polygon
    geographic coordinates of rectangle
  """

  lft = center.x - sz/2 # left border of the rectangle
  btm = center.y - sz/2 # bottom border of the rectangle

  aoi = ee.Geometry.Polygon(
           [[[lft, btm],
            [lft, btm+sz],
            [lft+sz, btm+sz],
            [lft+sz, btm]]], crs, False)
  return aoi

def cloudProbability(image: ee.Image, 
                     aoi: ee.Geometry.Polygon) -> ee.Image :
  """
  Specific for Sentinel-2 level 2A function.
  Calculate cloudness average probability in an area of interest on image

  Parameters
  ----------
  image : ee.Image
    image Sentinel-2 with layer MSK_CLDPRB (cloud probability, %)
  aoi : ee.Geometry.Polygon
    area of interest on image
  sz : float
    size of the rectangle side in a units of crs

  Returns
  -------
  ee.Image
    the image with added property 'cloudness' equals to average cloud probability in the area of interest
  """

  cloudness = (image.reduceRegion(ee.Reducer.sum(), 
                                  aoi, scale=10, 
                                  crs=aoi.projection())
                    .get('MSK_CLDPRB'))
  return image.set('cloudness', cloudness)


def get_sentinel2_tile(aoi: ee.Geometry.Polygon,
                       start_time: ee.Date,
                       finish_time: ee.Date,
                       cloudpercentage=15,
                       cloudbuffer=1e3,
                       tail_cloud_probability=5e2 # sum of cloud probability in each pixel
                       ) -> dict:
  """
  Creates numpy arrays for 10m and 20m bands from Sentinel-2 snapshots
  in area of interests filtered by dates and cloudness

  Parameters
  ----------
  aoi: ee.Geometry.Polygon
    Area of interest
  start_time: ee.Date
    The earliest date of snapshot
  finish_time: ee.Date
    The latest date of snapshot
    start_time--finish_time: date slot
  cloudpercentage : int, default 15
    The highest allowed CLOUD_COVERAGE_ASSESSMENT for Sentinel-2 snapshot
    Area percentage covered by clouds
  cloudbuffer : int, default 1000
    Distance from the area of interest included in tail_cloud_probability 
    calculation
  tail_cloud_probability : int default 500
    Sum of cloud probability in each pixel in the area of interest and some
    distance around. Calculated from mask MSK_CLDPRB in Sentinel-2 level 2A
  
  Returns
  -------
  dict : {'tile_10m', 'tile_20m', 'date', 'id'} or None
    tile_10m : numpy.array
      3D array from 10m bands ('B2', 'B3', 'B4', 'B8') of selected 
      Sentinel-2 snapshot from area of interest (height, width, bands)
    tile_20m : numpy.array
      3D array from 20m bands ('B5', 'B6', 'B7', 'B8A', 'B11', 'B12') 
      of selected Sentinel-2 snapshot from area of interest 
      (height, width, bands)
    date : datetime.datetime
      Date and time of Sentinel-2 snapshot
    id : str
      PRODUCT_ID of Sentinel-2 snapshot
    Returns None if no snapshots satisfying the conditions

  Exceptions
  ----------
  ee.EEException
    
  """

  proj = aoi.projection()
  # Image Collection of all Sentinel-2 snapshoots entirely covers AOI 
  # from start_time untill end_time with cloud coverge less than selected level
  tile = (ee.ImageCollection("COPERNICUS/S2_SR")
            .filterDate(start_time, finish_time)
            .filter(ee.Filter.lt('CLOUD_COVERAGE_ASSESSMENT', cloudpercentage))            
            .filterBounds(aoi)
            .filter(ee.Filter.contains('.geo', aoi)))
  # Pause for preventing ProtocolError: Connection aborted, 
  # ConnectionResetError(104, 'Connection reset by peer'))
  time.sleep(0.05)

  # add property <cloudness> to images and remove unnessesary bands
  # Cloudness is the sum of cloud probability for pixels in aoi and around it
  try : 
    withCloudness = (tile.select(['B2', 'B3', 'B4', 'B8', 
                                  'B5', 'B6', 'B7', 'B8A',
                                  'B11', 'B12', 'MSK_CLDPRB'])
                     .map(lambda im: cloudProbability(im, 
                                     aoi.buffer(cloudbuffer, 
                                                proj=proj)))
                     .sort('cloudness', True))
    time.sleep(0.05)
  except ee.EEException as e : 
    print('Error in tile selecting from Google Earth Engine', e)
    return None

  # cloudness of the image with least average cloud probability
  try :
    cloud_density = (withCloudness.first()
                                  #.sampleRectangle(region=aoi, defaultValue=0)
                                  .getNumber('cloudness')
                                  .getInfo())
    time.sleep(0.05)
  except ee.EEException as e : 
    print('Error in tile selecting from Google Earth Engine, high cloudness', e)
    return None

  # no tile generated for the date slot if any samples without clouds found
  if cloud_density > tail_cloud_probability :
    print('ATTENTION: cloud_density > tail_cloud_probability')
    return None
  # select bands with resolution 10 m
  visnir_bands_10m = (withCloudness.first()
                      .sampleRectangle(region=aoi, defaultValue=0)
                      .select(['B2', 'B3', 'B4', 'B8']))
  time.sleep(0.05)
  # select bands with resolution 20 m
  # buffer (-5 m) for exact 2 times less size in pixels than in 10 m bands
  visnir_bands_20m = (withCloudness.first()
                      .sampleRectangle(region=aoi.buffer(-5, proj=proj), 
                                       defaultValue=0)
                      .select(['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']))
  time.sleep(0.05)
  # ee.Image's to np.array's
  band_arr_b2 = np.array(visnir_bands_10m.get('B2').getInfo())
  time.sleep(0.05)
  band_arr_b3 = np.array(visnir_bands_10m.get('B3').getInfo())
  time.sleep(0.05)
  band_arr_b4 = np.array(visnir_bands_10m.get('B4').getInfo())
  time.sleep(0.05)
  band_arr_b8 = np.array(visnir_bands_10m.get('B8').getInfo())
  time.sleep(0.05)
  band_arr_b5 = np.array(visnir_bands_20m.get('B5').getInfo())
  time.sleep(0.05)
  band_arr_b6 = np.array(visnir_bands_20m.get('B6').getInfo())
  time.sleep(0.05)
  band_arr_b7 = np.array(visnir_bands_20m.get('B7').getInfo())
  time.sleep(0.05)
  band_arr_b8a = np.array(visnir_bands_20m.get('B8A').getInfo())
  time.sleep(0.05)
  band_arr_b11 = np.array(visnir_bands_20m.get('B11').getInfo())
  time.sleep(0.05)
  band_arr_b12 = np.array(visnir_bands_20m.get('B12').getInfo())
  # construct 3D array [band, x, y] for 10m and 20m bands
  arr_10m = np.dstack((band_arr_b2, band_arr_b3, band_arr_b4, band_arr_b8))
  arr_10m = np.moveaxis(arr_10m, [0, 1, 2], [2, 1, 0])
  arr_20m = np.dstack((band_arr_b5, band_arr_b6, band_arr_b7, 
                       band_arr_b8a, band_arr_b11, band_arr_b12))
  arr_20m = np.moveaxis(arr_20m, [0, 1, 2], [2, 1, 0])
  # extract time of the shoot
  tile_time = (datetime.fromtimestamp(int(withCloudness
                       .first().get('system:time_start')
                       .getInfo())/1000))
  # extract index and tile ID
  tile_index = withCloudness.first().get('system:index').getInfo()
  product_id = withCloudness.first().get('PRODUCT_ID').getInfo()
  print(f'Cloud density on the tail: {cloud_density}\nIndex of the tile: {tile_index}\nTime of shoot: {tile_time}\nID: {product_id}')
  if band_arr_b2.mean() > 1000 :
    print('ATTENTION: CLOUD COVER HIGHLY PROBABLY UNFILTERED!')
    print(f'Means of B2, B3, B4, B8 bands: {band_arr_b2.mean()}, {band_arr_b3.mean()}, {band_arr_b4.mean()}, {band_arr_b8a.mean()}')
  
  return {'tile_10m': arr_10m, 'tile_20m': arr_20m, 'date': tile_time, 'id': product_id}

def extract_sentinel2_inside_geodataframe(data_collector: satteliteTiles,
                                          areas: gpd.GeoDataFrame, 
                                          dates: list, # of datetime.datetime [start, finish]
                                          buff=20, square=320,
                                          columns=[] # list of str
                                         ):
  """
  Collects tile arrays for random site inside parcels described in <areas>
  for dates inside slotes described in <dates>
 
  Parameters
  ----------
  data_collector: satteliteTiles
    object of satteliteTiles class for manipulations with
    files of tile arrays and context information in pandas.DataFrame format
  areas: gpd.GeoDataFrame
    parcels for tiles search in
  dates: list of datetime.datetime pairs
    intervals of dates for tiles search in;  [[start, finish], ...]
  buff=20
    minimal distance from a border of a parcel for center of a tile 
    in units of crs (m)
  square=320
    lenght of tile side (square) in units of crs (m)
  columns=[] : list of str
    names of variables with context information in <areas> to copy in <data_collector>

  Returns
  -------
  The function updates DataFrame connected with <data_collector> 
  by information about collected tiles and save tile arrays in .npy files
  """
  # finding random point in one of the compartments in <areas>
  # sampling is (coordinates, compartment geoDataSeries)
  sampling = extract_point_in_compartment(areas, buff=buff) 
  center = sampling[0] # x, y coordinates of the point

  # crs of areas and its conversion into ee.Projection type
  source_crs = areas.geometry.crs
  crs_ee = ee.Projection(f'EPSG:{source_crs.to_authority()[1]}')

  # defining square area of interest around point
  # for bands with spatial resolution 10m we have to reduse 
  # sides of the square on 10 m for 10*size_in_pixels == size_in_meters
  aoi = get_aoi(center, crs_ee, sz=square-10)
  print(f'The area of interest is {aoi.getInfo()}')
  point_geohash = get_geohash(center, source_crs)

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
    sampling[1]['product_id'] = tiledata['id']
    data_collector.add_record(tiledata['tile_10m'], 10.0, tiledata['date'], 
                              point_geohash, 
                              sampling[1][columns + ['product_id']], 
                              '10m02030408')
    data_collector.add_record(tiledata['tile_20m'], 20.0, tiledata['date'], 
                              point_geohash, 
                              sampling[1][columns + ['product_id']], 
                              '20m050607081112')
