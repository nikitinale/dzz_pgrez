import os
import geopandas as gpd

# default directory and shapefile of forest inventory data with shapefiles
# Polessie State Radiation and Ecology Reserve
# Projection (crs) of the shapefile is EPSG:32635
basedir = os.path.expanduser('~/mybox/IRB/lespgrez/pgrez_gis')
shapefile = '330_subcompartments_region.shp'

def load_forest_geodatabase(basedir=basedir, 
                            shapefile=shapefile) -> gpd.GeoDataFrame:
  """
  Open shapefile with forest inventory data in format of geoDataFrame

  Parameters
  ----------
  basedir : str
    directory with shapefile
  shapefile : str
    name of shapefile

  Returns
  -------
  geopandas.GeoDataFrame
    geometries and forest inventory information
  """  
  vydela = gpd.read_file(os.path.join(basedir, shapefile))
  return vydela
