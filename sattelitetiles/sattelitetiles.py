import os
import numpy as np
import pandas as pd
from datetime import datetime


class satteliteTiles():
    """
    Class for keeping Sentinel tiles with context information
    Tiles are in numpy .npy files
    Context information is in pandas.DataFrame tables
    Geohash is used in .npy file names and in DataFrame 
    for the fast search of the data
    """

    def __init__(self, datafolder: str):
        """
        Initialization of the Sentinel tiles and context data

        Parameters
        ----------
        datafolder : str
          directory for keeping Sentinel tiles (.npy files)
          and context information in a file description.csv
        """

        # folder with tiles and information about them
        self.datafolder = datafolder
        # name of csv file with information about tiles
        self.csvfile = 'description.csv'
        new = not os.path.exists(os.path.join(datafolder, self.csvfile))
        if new:
            os.makedirs(self.datafolder, exist_ok=True)
            # dataframe is not exist
            self.df = None
        else:
            self.df = pd.read_csv(os.path.join(self.datafolder, self.csvfile),
                                  index_col=[0])

    def add_record(self,
                   tile: np.array,
                   resolution: float, # m/px
                   date: datetime,
                   centroid: str, # geohash
                   fields: dict,
                   productname: str) :
        """"
        Adding new data into Sentinel Dataset for machine learning

        Parameters
        ----------
        tile: np.array
          numpy array [channels, height, width] with tile from a satellite shoot
        resolution: float
          resolution of the sattelite shoot in m/px
          For Sentinel-2 it is equal 10, 20 and 60 m/px
        date: datetime
          date and time of the shoot
        centroid: str
          geohash of the sattelite shoot centrum
          Used in file name
        fields: dict
          dictionaty with fields names (columns names) and values 
          with context information about the sattelite shoot
        productname: str
          Codename of the source product for identification of tile in the dataset

        Returns
        -------
        True : if new data added into dataset
        none : if data did not added into dataset
        """             

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
        if isinstance(self.df, pd.DataFrame) :
            self.df = self.df.append(fields, ignore_index=True)
        else :
            self.df = pd.DataFrame([fields])
        self.df.to_csv(os.path.join(self.datafolder, self.csvfile))
    
        # saving numpy array on drive
        np.save(os.path.join(self.datafolder, fields['filename']), tile)
        return True

    def dataset_initialize():
        self.initialized = False
        if not self.df:
            return self.initialized
