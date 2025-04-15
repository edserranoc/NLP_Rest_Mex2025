import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import re, unicodedata
from typing import List, Dict
import matplotlib.ticker as mticker

class load_data:
    def __init__(self):
        self.path_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.path_data = os.path.join(self.path_project, 
                                      'data',
                                      'Train_DataSet',
                                      'Rest-Mex_2025_train.csv')  # Path to the data file
        self.states_map = os.path.join(self.path_project, 'data', 'maps', 'ent.gpkg')
        self.mun_map = os.path.join(self.path_project, 'data', 'maps', 'mun.gpkg')
        
        self.data = self.read_data()
        self.states_map = self.read_states_map()
        self.mun_map = self.read_mun_map()
        self.data["Region"] = self.data["Region"].apply(lambda x: self.normalize_state(x))
        self.states_map["NOMGEO"] = self.states_map["NOMGEO"].apply(lambda x: self.normalize_state(x))
        
    def read_data(self):
        return pd.read_csv(self.path_data, sep=',')

    def normalize_state(self,name):
        """
        Normalize the state name
        """
        name = name.lower()
        name = unicodedata.normalize("NFD", name)
        name = re.sub(r'[\u0300-\u036f]', '', name)
        name = re.sub(r'\s+|_', '', name)
        name = re.sub(r'[^\w]', '', name) 
        if name == "mexico":
            name = "estadodemexico"
        if name == "michoacandeocampo":
            name = "michoacan"
        if name == "veracruzdeignaciodelallave":
            name = "veracruz"
        if name == "coahuiladezaragoza":
            name = "coahuila"
        return name
    
    def read_states_map(self):
        return gpd.read_file(self.states_map,columns=["NOMGEO","geometry"])
    
    def read_mun_map(self):
        return gpd.read_file(self.mun_map,columns=["NOMGEO","geometry"])

    def merge_states(self):
        """
        Merge the data with the states map
        """
        # Merge the data with the states map
        data = self.states_map.copy()
        data["NOMGEO"] = data["NOMGEO"].apply(lambda x: self.normalize_state(x))
        data = data.rename(columns={"NOMGEO": "Region"})
        
        # Merge the groupby count data of self.data with the states map
        number_of_reviews = self.data.groupby("Region").size().reset_index(name='counts')
        
        data['counts'] = data['Region'].map(number_of_reviews.set_index('Region')['counts'])
        #data['counts'] = data['counts'].fillna(NaN)
        
        return data
    
    def plot_states(self,   params:Dict= None,
                            params_bar:Dict= None,
                            save:bool=False):
        """
        Plot the states map
        """
        # Plot number of reviews by state
        data = self.merge_states()
        
        if params is None:
            params = { "column": "counts",
                        "cmap": "Purples",
                        "legend": True,
                        "edgecolor": 'black',
                        "missing_kwds": {'color': 'white','linewidth': 0.1},
                        "figsize": (20,6),
                        "linewidth": 0.1,
                        "legend_kwds": {
                            "shrink": 0.2,
                            "orientation": "horizontal",  # or "horizontal"
                            "aspect": 85,
                            "pad": 0.1,                           
                            }
                        }
            
        if params_bar is None:
            params_bar = {'label': 'Number of Reviews by State',
                            'loc': 'left',
                            'fontdict': {'family': 'serif',
                                         'size': 10,      # optional
                                         'color': 'black'
                                     }
                          }

        fig, ax = plt.subplots(figsize=(20, 6))
        
        # Plot the states map
        data.plot(**params,ax=ax)
        

        cbar = ax.get_figure().get_axes()[-1]
        #cbar.set_ylabel('Number of reviews')
        cbar.set_title(**params_bar)
        
        ax.set_axis_off()
        plt.show()
        
        
if __name__ == "__main__":
    bf = load_data()
    data = bf.read_data()
    states_map = bf.read_states_map()
    print(states_map.head())
    print(data.head())

