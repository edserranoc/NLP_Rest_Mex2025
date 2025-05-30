import os
import pandas as pd
from typing import List, Tuple
import numpy as np
import geopandas as gpd

from src.pipelines.data_loader import DataLoader2, ReviewLabelExtractor
from src.pipelines.dataset import DatasetSplitter
from src.pipelines.preprocessor import TextPreprocessor, EmbeddingLoader
from src.visualization.plot_generator import PlotGenerator, TextFeatureAnalyzer, MapPlotter


class DataPipeline:
    def __init__(self, device: str, stopwords: set = None, max_len: int = None):
        try:
            root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
            self.data_path = os.path.join(root_path, 'data', 'Train_DataSet', 'Rest-Mex_2025_train.csv')
            self.emb_path = os.path.join(root_path, 'data', 'embeddings', 'word2vec_col.txt')
        except FileNotFoundError:
            self.data_path = '/content/Rest-Mex_2025_train.csv'
            self.emb_path = '/content/word2vec_col.txt'

        self.df = DataLoader2(self.data_path).get_data()

        self.corpus, self.labels = ReviewLabelExtractor(self.df).extract()

        self.processor = TextPreprocessor(stopwords = stopwords)

        self.vocab, self.emb_mat = EmbeddingLoader(self.emb_path).load(device = device)

        # Diccionarios para mapear Type y Town a índices
        self.type2id = {label: idx for idx, label in enumerate(sorted(self.df['Type'].dropna().unique()))}
        self.id2type = {v: k for k, v in self.type2id.items()}

        self.town2id = {label: idx for idx, label in enumerate(sorted(self.df['Town'].dropna().unique()))}
        self.id2town = {v: k for k, v in self.town2id.items()}

        if max_len is None:
            token_lengths = [len(self.processor.process(r)) for r in self.corpus]
            max_len = int(np.percentile(token_lengths, 99))
            print(f"max_len = {max_len} (percentil 99)")

        # Split
        splitter = DatasetSplitter(self.corpus, self.labels)
        self.train_dataset, self.val_dataset = splitter.stratified_split(
            vocab = self.vocab,
            processor = self.processor,
            max_len = max_len,
            device = device
        )
    

class DataPipelineVisualization:
    """
    DataPipelineVisualization class for visualizing data from the DataPipeline.
    This class inherits from DataPipeline and provides methods to visualize the data.
    """
    def __init__(self):
        try:
            root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
            self.data_path = os.path.join(root_path, 'data', 'Train_DataSet', 'Rest-Mex_2025_train.csv')
            self.emb_path = os.path.join(root_path, 'data', 'embeddings', 'word2vec_col.txt')
            self.states_map = os.path.join(root_path, 'data', 'maps', 'ent.gpkg')
        except FileNotFoundError:
            self.data_path = '/content/Rest-Mex_2025_train.csv'
            self.emb_path = '/content/word2vec_col.txt'

        self.df = DataLoader2(self.data_path).get_data()
        self.states_map = gpd.read_file(self.states_map,columns=["NOMGEO","geometry"])
        self.corpus, self.labels = ReviewLabelExtractor(self.df).extract()
    
        self.plotter = PlotGenerator(self.df)
        self.text_analyzer = TextFeatureAnalyzer(self.corpus, self.labels)
        self.map_plotter = MapPlotter(self.df, self.states_map)
        
    def plot_sentiment_by_region(self, ax = None): return self.plotter.plot_sentiment_by_region(ax)
    def plot_sentiment_by_town(self, ax = None): return self.plotter.plot_sentiment_by_town(ax)
    def plot_polarity_histogram(self, ax = None): return self.plotter.plot_polarity_histogram(ax)
    def plot_polarity_pie(self, ax = None): return self.plotter.plot_polarity_pie(ax)
    def plot_region_histogram(self, ax = None): return self.plotter.plot_region_histogram(ax)
    def plot_town_histogram(self, ax = None): return self.plotter.plot_town_histogram(ax)
    def plot_sentiment_histogram(self, ax = None): return self.plotter.plot_sentiment_histogram(ax)

    def plot_general_wordcloud(self, k: int, method: str, label_type: str):
        return self.text_analyzer.plot_general_wordcloud(k = k, method = method, label_type = label_type)

    def plot_class_representative_wordcloud(self, class_label: str, label_type: str, k: int):
        return self.text_analyzer.plot_class_representative_wordcloud(class_label = class_label, label_type = label_type, k = k)
    
    def plot_map_states(self, ax = None):
        return self.map_plotter.plot_map(ax)