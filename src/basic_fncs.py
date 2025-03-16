import numpy as np
import pandas as pd
import geopandas as gpd

class BasicFunctions:
    def __init__(self):
        self.data_folder = '../data'
        self.data_path = self.data_folder+'/Rest-Mex_2025_Train_DataSet'
        self.states_map = self.data_folder+'/mxmaps/ent.gpkg'
        
    
    def read_data(self, path=None):
        if path is None:
            path = self.data_path
        return pd.read_csv(path)
    
    def read_states_map(self, path=None):
        if path is None:
            path = self.states_map
        return gpd.read_file(path)
    
        