import numpy as np
import pandas as pd
import geopandas as gpd
import os

class load_data:
    def __init__(self):
        self.path_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path_data = os.path.join(self.path_project, 'data','Rest-Mex_2025_Train_DataSet','Rest-Mex_2025_train.csv')  # Path to the data file
        self.states_map = os.path.join(self.path_project, 'data', 'maps', 'ent.gpkg')
        self.mun_map = os.path.join(self.path_project, 'data', 'maps', 'mun.gpkg')
        
    def read_data(self):
        return pd.read_csv(self.path_data, sep=',')

    
    def read_states_map(self):
        return gpd.read_file(self.states_map)[["NOMGEO","geometry"]]
    
    def read_mun_map(self):
        return gpd.read_file(self.mun_map)[["NOMGEO","geometry"]]

if __name__ == "__main__":
    bf = load_data()
    data = bf.read_data()
    states_map = bf.read_states_map()
    print(states_map.head())
    print(data.head())
        