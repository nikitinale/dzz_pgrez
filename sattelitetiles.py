import os
import numpy as np
import pandas as pd
from datetime import datetime

class satteliteTiles :

  def __init__(self, datafolder: str) :
    # folder with tiles and information about them
    self.datafolder = datafolder
    # name of csv file with information about tiles
    self.csvfile = 'description.csv'
    new = not os.path.exists(os.path.join(datafolder, self.csvfile))
    if new :
      os.makedirs(self.datafolder, exist_ok=True)
      # dataframe is not exist
      self.df = None
    else :
      self.df = pd.read_csv(os.path.join(self.datafolder, self.csvfile), index_col=[0])

  def add_record(self,
                 tile: np.array,
                 resolution: float, # m/px
                 date: datetime,
                 centroid: str, # geohash
                 fields: dict,
                 productname: str) :
                 
    # first part of record name is centroid geohash
    # second part of record name indicating date
    datename = date.strftime('%Y%m%d')

    # composing information about tile
    fields['geohash'] = centroid
    fields['date'] = date
    fields['resolution'] = resolution
    fields['product'] = productname
    fields['filename'] = '_'.join([productname, centroid, datename])

    # saving information about tile in DataFrame on drive
    #try :
    if isinstance(self.df, pd.DataFrame) :
      self.df = self.df.append(fields, ignore_index=True)
    else :
      self.df = pd.DataFrame([fields])
    self.df.to_csv(os.path.join(self.datafolder, self.csvfile))
    #except :
    #  return None
    
    # saving numpy array on drive
    #try :
    np.save(os.path.join(self.datafolder, fields['filename']), tile)
    #except :
    #  return None
    return True
