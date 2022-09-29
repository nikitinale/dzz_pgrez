# dzz_pgrez
Remote sensing Sentinel-2 for forest stands condition forecasting in the Polessie State Radiation and Ecology Reserve



**dzz_pgrez**

- sattelitetiles: Functions and classes for retrieving fragments of satellite Earth observation from Google Earth Engine for areas of interest.
  - extractsdotcompartment.py -- functions for sampling random or grided points in areas of interest.
  - sattelitetiles.py -- Class for keeping Sentinel fragments (numpy.array) with context information (pandas.DataFrame) for machine learning.
  - geeloopsentineltiles.py -- functions for retrieving square fragments of Sentinel-2 data from Google Earth Engine. Function extract_sentinel2_inside_geodataframe creates collection of the Sentinel-2 fragments (see class satteliteTiles) for machine learning.
  - point_tiles.py -- Function for retrieving square fragments of Sentinel-2 data from Google Earth Engine for selected time slotes
  - initialize_ee.ipynb -- Jupyter notebook for activation of Google Earth Engine account. Must be used before calling any Google Earth Engine functions.

