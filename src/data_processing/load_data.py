import numpy as np
import pandas as pd
import geopandas as gpd
import os
import re, unicodedata

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
    
    def plot_states_map(self, save=False):
        """
        Plot the states map
        """
        # Plot number of reviews by state
        data = self.merge_states()
        
        # Plot the states map
        ax = data.plot(column="counts", 
                                  cmap="YlGnBu", 
                                  legend=True,
                                  edgecolor='black',
                                  missing_kwds={'color': 'white','linewidth': 0.07},
                                  figsize=(18,6),
                                  linewidth=0.1
                                  )

        ax.set_title("Number of reviews by state")
        ax.set_axis_off()
        
    
if __name__ == "__main__":
    bf = load_data()
    data = bf.read_data()
    states_map = bf.read_states_map()
    print(states_map.head())
    print(data.head())

