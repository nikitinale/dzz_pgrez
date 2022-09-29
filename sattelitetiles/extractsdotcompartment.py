import math
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import numpy as np
from itertools import product
#from random import uniform
import geopandas as gpd
import geohash
import pyproj


def samplig_points_from_area(area: Polygon,
                             count: int,
                             maxiter=0,
                             crs='EPSG:4326') -> gpd.GeoDataFrame:
    """
    Sampling random points from defined area.
    First step: generate random point in 2D bounding box of the area
    Second step: check is the point lies inside the area; if not -- repeat 

    Parameters
    ----------
    area : shapely.geometry.Polygon 
      defines area of sampling
    count : int
      number of points sampled from the area
    maxiter : int, default value 0
      the maximum number of attempts to find random points inside the area. 
      If maxiter==0, the number of attempts is equal to count*100
    crs : str, 'EPSG:4326' by default
      crs of geoDataFrame with sampled points.

    Returns
    -------
    geoDataFrame
      geographic coordinates of found random points from define area
      The number of returned points may be less than <count> 
      if <maxiter> achieved early than requared number of poins are found
    """

    countin = 0  # how many points were found
    curiter = 0  # how many attempts were used
    # default number of attempts
    if maxiter < 1:
        maxiter = count*100
    pointsin = []  # list with coordinates of found points
    # make search untill maximum number of attempts or defined number of found points
    while((countin < count) and (curiter < maxiter)):
        curiter += 1
        # find random point in the bounding box of the area
        x = np.random.uniform(area.bounds[0], area.bounds[2])
        y = np.random.uniform(area.bounds[1], area.bounds[3])
        sample_point = Point(x, y)
        # check is the random point iside the area
        if sample_point.within(area):
            countin += 1
            pointsin.append(sample_point)

    # create geoDataFrame from the found points
    sampledot = gpd.GeoDataFrame({'geometry': pointsin, }, crs=crs)
    return sampledot


def sampledot_from_area(area: Polygon,
                        maxiter=1000) -> Point:
    """
    Sampling a random point from defined area.
    First step: generate random point in 2D bounding box of the area
    Second step: check is the point lies inside the area; if not -- repeat 

    Parameters
    ----------
    area : shapely.geometry.Polygon 
      defines area of sampling
    maxiter : int, default value 1000
      the maximum number of attempts to find random point inside the area

    Returns
    -------
    shapely.geometry.Point
      coordinates of random points from define area
      if any point found, returns None
    """
    curiter = 0       # how many points were found
    sample_point = None  # coordinates of found point
    # make search untill maximum number of attempts or found point
    while(curiter < maxiter):
        curiter += 1
        # find random point in the bounding box of the area
        x = np.random.uniform(area.bounds[0], area.bounds[2])
        y = np.random.uniform(area.bounds[1], area.bounds[3])
        sample_point = Point(x, y)
        # check is the random point iside the area
        if sample_point.within(area):
            return sample_point
    # return default point if not found -- None
    return sample_point


def extract_point_in_compartment(areas: gpd.GeoDataFrame,
                                 maxiter=1000,
                                 buff=20) -> (Point, gpd.GeoSeries):
    """
    Sampling a random point inside random compartment in 
    a defined distance from border of the compartmint

    Parameters
    ----------
    areas : geoDataFrame
      geodatabese with areas of compartments in a column "geometry"
    maxiter : int, default 1000
      the maximum number of attempts to find random point inside a compartmint
    buff : float, default 20
      minimal distance from a border of a compartment for sampling a random point in units of crs
      positive -- directon inside the compartment, negative -- directon outside the compartment

    Returns
    -------
    (shapely.geometry.Point, gpd.geoDataSeries)
      Point : coordinates of sampled point in crs of <areas>
      geoDataSeries : a row from areas dataframe which contains the point 
        if any poind found during <maxiter> iteration returns (None, random geoDataSeries)
    """

    curiter = 0       # how many attempts were used
    extracted_point = None  # coordinates of a sampled point
    while curiter < maxiter:
        curiter += 1
        # random compartment from geoDataFrame
        vydel = areas.sample(1, axis=0).iloc[0]
        # limiting area in the compartment <buff> from border
        uchastok = vydel.geometry.buffer(-1*buff)
        if uchastok.area > 0:  # the limited area could be zero
            # finding random point in the limited area
            extracted_point = sampledot_from_area(
                uchastok, maxiter=maxiter)  # (x, y)
            return extracted_point, vydel
    return extracted_point, vydel

def sampling_greed_in_compartments(areas: gpd.GeoDataFrame,
                                   step: int=10,
                                   padding: float=0,
                                   maxiter: int=1000,
                                   buff: float=0) -> gpd.GeoDataFrame:
    """ Sampling set of points in nodes of a greed with size <step> inside all
        compartments <areas> in a defined distance from border of the compartments

    Parameters
    ----------
    areas : geoDataFrame
      geodatabese with areas of compartments in a column "geometry"
    step: int, default 10
      Size of the greed cells (distance between the points in a generated set)
      in units of crs
    padding: float, default 0
      paddig from a sides of a common rectangle bound of all areas in units of crs
    maxiter : int, default 1000
      the maximum number of attempts to find random point inside a compartmint
    buff : float, default 20
      minimal distance from a border of a compartment for sampling a random
      point in units of crs
      positive -- directon inside the compartment, negative -- 
      directon outside the compartment

    Returns : gpd.geoDataFrame
    -------
      DataFrame with column "points" contains coordinates of sampled set of
      points and corresponding supplimentary data from DataFrame <areas> in
      additional columns
    """

    _areas = areas.geometry.unary_union
    bounds = _areas.bounds
    xstart = math.floor(bounds[0]+padding)
    ystart = math.floor(bounds[1]+padding)
    xap = bounds[0] + padding - xstart
    yap = bounds[1] + padding - ystart
    xx = [x+xap for x in range(xstart, math.ceil(bounds[2]-padding+step), step)]
    yy = [y+yap for y in range(ystart, math.ceil(bounds[3]-padding+step), step)]
    xy = [Point(x, y) for x, y in product(xx, yy)]
    points_df = gpd.GeoDataFrame({'points': xy,}, geometry='points', crs=areas.crs)
    points_df = gpd.sjoin(points_df, areas)
    return(points_df)
