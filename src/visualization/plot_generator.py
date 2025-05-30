import pandas as pd
import geopandas as gpd
import numpy as np
import re, unicodedata

from typing import List, Tuple
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.feature_selection import mutual_info_classif, chi2
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colormaps
import seaborn as sns


class NameFormatter:
    
    @staticmethod
    def name_formatter(name: str) -> str:
        name_map = {
            'QuintanaRoo'        : 'Quintana Roo',
            'Michoacan'          : 'Michoacán',
            'Yucatan'            : 'Yucatán',
            'Queretaro'          : 'Querétaro',
            'Baja_CaliforniaSur' : 'Baja California Sur',
            'San_Luis_Potosi'    : 'San Luis Potosí',
            'Estado_de_Mexico'   : 'Estado de México',
            'TodosSantos'        : 'Todos Santos',
            'Patzcuaro'          : 'Pátzcuaro',
            'Teotihuacan'        : 'Teotihuacán',
            'Tepoztlan'          : 'Tepoztlán',
            'Cuatro_Cienegas'    : 'Cuatro Ciénegas',
            'Tepotzotlan'        : 'Tepotzotlán',
            'Zacatlan'           : 'Zacatlán',
        }
        if name in name_map:
            return name_map[name]
        elif '_' in name:
            return name.replace('_', ' ').title()        
        else:
            return name



class PlotGenerator:
    """
    PlotGenerator class for generating various plots from a DataFrame. This class takes a DataFrame as input
    and provides methods to generate different types of plots. The plots include histograms, pie charts,
    bar charts, and more. The plots can be customized with different colors and styles.
    """
    def __init__(self, df: pd.DataFrame):
        self.data = df

    def plot_polarity_histogram(self, ax = None) -> None:
        """
        Plots a histogram of sentiment polarity scores.
        """
        counts = self.data['Polarity'].value_counts().sort_index()
        if ax is None:
            fig, ax = plt.subplots(figsize = (8,5))
        bars = ax.bar(counts.index, counts.values, color = sns.color_palette("YlGnBu", 5), edgecolor = 'black')
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy = (bar.get_x() + bar.get_width() / 2, height),
                        xytext = (0,4),
                        textcoords = "offset points",
                        ha = 'center', va = 'bottom', fontsize = 10)
        ax.set_title('Polarity Score Distribution', fontsize = 14, fontweight = 'bold')
        ax.set_ylabel('Number of Reviews', fontsize = 12)
        ax.set_xlabel('Polarity Score', fontsize = 12)
        ax.tick_params(axis = 'x', rotation = 0)
        ax.grid(axis = 'y', linestyle = '--', alpha = 0.3)

    def plot_polarity_pie(self, ax = None) -> None:
        """
        Plots a pie chart of sentiment polarity distribution.
        """
        counts = self.data['Polarity'].value_counts().sort_index()
        labels = [str(int(p)) for p in counts.index]
        sizes = counts.values
        if ax is None:
            fig, ax = plt.subplots(figsize = (7,7))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels = labels,
            explode = (0, 0, 0, 0, 0.1),
            colors = sns.color_palette("viridis", len(labels)),
            autopct = '%1.1f%%',
            startangle = 90,
            textprops = {'fontsize':11},
            pctdistance = 0.85,
            labeldistance = 1.05,
            wedgeprops = {'edgecolor':'black'}
        )
        for text in texts: # outer labels
            text.set_color('black')
            text.set_fontweight('bold')
            text.set_fontsize(10)
        for autotext in autotexts: # percentages
            autotext.set_color('lightgray')
        for w in wedges:
            w.set_linewidth(1)
            w.set_edgecolor('black')
        ax.set_title('Polarity Score Distribution', fontsize = 14, fontweight = 'bold')
        ax.axis('equal') 

    def plot_sentiment_histogram(self, ax = None) -> None:
        """
        Plots a bar chart of sentiment categories (Positive, Neutral, Negative).
        """
        counts = self.data['Sentiment'].value_counts()
        counts = counts.reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
        if ax is None:
            fig, ax = plt.subplots(figsize = (8,5))
        bars = ax.bar(counts.index, counts.values, color = ['#4CAF50', '#9E9E9E', '#F44336'], edgecolor = 'black')
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy = (bar.get_x() + bar.get_width() / 2, height),
                        xytext = (0,4),
                        textcoords = "offset points",
                        ha = 'center', va = 'bottom', fontsize = 10)
        ax.set_title('Sentiment Distribution', fontsize = 14, fontweight = 'bold')
        ax.set_ylabel('Number of Reviews', fontsize = 12)
        ax.set_xlabel('Sentiment Category', fontsize = 12)
        ax.tick_params(axis = 'x', rotation = 0)
        ax.grid(axis = 'y', linestyle = '--', alpha = 0.3)

    def plot_region_histogram(self, ax = None) -> None:
        """
        Plots the number of reviews per region.
        """
        counts = self.data['Region'].value_counts().sort_values(ascending = False)
        formal_labels = [NameFormatter.name_formatter(region) for region in counts.index]
        cmap = colormaps['Blues']
        norm = LogNorm(vmin = max(1, min(counts.values)), vmax = max(counts.values))
        if ax is None:
            fig, ax = plt.subplots(figsize = (10,8))
        colors = [cmap(norm(v)) for v in counts.values]
        bars = ax.barh(formal_labels, counts.values, color = colors, edgecolor = 'black')
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{int(width)}',
                        xy = (width, bar.get_y() + bar.get_height() / 2),
                        xytext = (5,0), textcoords = "offset points",
                        ha = 'left', va = 'center', fontsize = 9, fontweight = 'bold')
        ax.set_title('Reviews by Region', fontsize = 14, fontweight = 'bold')
        ax.set_xlabel('Number of Reviews', fontsize = 12)
        ax.grid(axis = 'x', linestyle = '--', alpha = 0.4)
        ax.set_ylim(-0.5, len(formal_labels) - 0.5)
        ax.invert_yaxis()

    def plot_sentiment_by_region(self, ax = None) -> None:
        """
        Plots the sentiment distribution by region.
        """
        df = self.data.copy()
        grouped = df.groupby(['Region', 'Sentiment']).size().unstack(fill_value = 0)
        grouped = grouped.loc[grouped.sum(axis = 1).sort_values(ascending = False).index]
        formal_labels = [NameFormatter.name_formatter(region) for region in grouped.index]
        if ax is None:
            fig, ax = plt.subplots(figsize = (10,8))
        y_pos = np.arange(len(grouped))
        left = np.zeros(len(grouped))
        for sentiment, color in zip(['Negative', 'Neutral', 'Positive'], ['#F44336', '#9E9E9E', '#4CAF50']):
            values = grouped[sentiment].values
            bars = ax.barh(y_pos, values, left = left, label = sentiment,
                           color = color, edgecolor = 'black')
            left += values
        ax.set_yticks(y_pos)
        ax.set_yticklabels(formal_labels)
        ax.set_title('Sentiment Distribution by Region', fontsize = 14, fontweight = 'bold')
        ax.set_xlabel('Number of Reviews', fontsize = 12)
        ax.grid(axis = 'x', linestyle = '--', alpha = 0.3)
        ax.set_ylim(-0.5, len(formal_labels) - 0.5)
        ax.invert_yaxis()
        ax.legend(loc = 'lower right')

    def plot_town_histogram(self, ax = None) -> None:
        """
        Plots the number of reviews per town.
        """
        counts = self.data['Town'].value_counts().sort_values(ascending = False)
        formal_labels = [NameFormatter.name_formatter(town) for town in counts.index]
        cmap = colormaps['YlGnBu']
        norm = LogNorm(vmin = max(1, min(counts.values)), vmax = max(counts.values))
        if ax is None:
            fig_height = max(6, len(formal_labels) * 0.3)
            fig, ax = plt.subplots(figsize = (10,fig_height))
        colors = [cmap(norm(v)) for v in counts.values]
        bars = ax.barh(formal_labels, counts.values, color = colors, edgecolor = 'black')
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{int(width)}',
                        xy = (width, bar.get_y() + bar.get_height() / 2),
                        xytext = (3.5,0), textcoords = "offset points",
                        ha = 'left', va = 'center', fontsize = 8)
        ax.set_title('Reviews by Town', fontsize = 14, fontweight = 'bold')
        ax.set_xlabel('Number of Reviews', fontsize = 12)
        ax.grid(axis = 'x', linestyle = '--', alpha = 0.4)
        ax.set_ylim(-0.5, len(formal_labels) - 0.5)
        ax.invert_yaxis()

    def plot_sentiment_by_town(self, ax = None) -> None:
        """
        Plots the sentiment distribution by town.
        """
        df = self.data.copy()
        grouped = df.groupby(['Town', 'Sentiment']).size().unstack(fill_value = 0)
        grouped = grouped.loc[grouped.sum(axis = 1).sort_values(ascending = False).index]
        formal_labels = [NameFormatter.name_formatter(town) for town in grouped.index]
        if ax is None:
            fig_height = max(6, len(formal_labels) * 0.3)
            fig, ax = plt.subplots(figsize = (10,fig_height))
        y_pos = np.arange(len(grouped))
        left = np.zeros(len(grouped))
        for sentiment, color in zip(['Negative', 'Neutral', 'Positive'], ['#F44336', '#9E9E9E', '#4CAF50']):
            values = grouped[sentiment].values
            bars = ax.barh(y_pos, values, left = left, label = sentiment,
                           color = color, edgecolor = 'black')
            left += values
        ax.set_yticks(y_pos)
        ax.set_yticklabels(formal_labels)
        ax.set_title('Sentiment Distribution by Town', fontsize = 14, fontweight = 'bold')
        ax.set_xlabel('Number of Reviews', fontsize = 12)
        ax.grid(axis = 'x', linestyle = '--', alpha = 0.3)
        ax.set_ylim(-0.5, len(formal_labels) - 0.5)
        ax.invert_yaxis()
        ax.legend(loc = 'center right', fontsize = 16, labelspacing = 1.5, handlelength = 2.5)
        

class TextFeatureAnalyzer:
    """
    TextFeatureAnalyzer class for analyzing text features and generating word clouds. This class takes a
    list of reviews and a list of labels as input. The reviews are tokenized and the features are extracted
    using different methods such as mutual information, chi-squared, normalized mutual information, and log odds.
    """
    def __init__(self, corpus: List[str], labels: List[Tuple[float, str, str]]):
        """
        Initialize the TextFeatureAnalyzer with a list of reviews and labels.
        """
        self.corpus = corpus
        self.labels = labels

    def plot_general_wordcloud(self, method: str, label_type: str, k: int = 50):
        """
        Generates an informative word cloud for the entire corpus, highlighting words that best discriminate
        between classes (based on label_type).
        
        Parameters
        ----------
        method : str
            Method to use for feature extraction. Options are: 
            - 'mi'   : Mutual Information.
            - 'chi2' : Chi-Squared.
            - 'nmi'  : Normalized Mutual Information.
        label_type : str
            Type of label to use for feature extraction. Options are:
            - 'Polarity'
            - 'Town'
            - 'Type'
        k : int
            Number of top words to include in the word cloud.
        """
        if label_type == 'Polarity':
            y = [int(p[0]) for p in self.labels]
        elif label_type == 'Town':
            y = [p[1] for p in self.labels]
        elif label_type == 'Type':
            y = [p[2] for p in self.labels]
        else:
            raise ValueError("label_type must be 'Polarity', 'Town' or 'Type'")
        vectorizer = CountVectorizer(max_features = 5000)  # Create a CountVectorizer instance to convert text to a matrix of token counts.
        X = vectorizer.fit_transform(self.corpus)          # Transform the corpus into a bag-of-words representation (n_documents x n_features).
        feature_names = vectorizer.get_feature_names_out() # Contains the words mapping to the columns of the matrix.
        word_scores = {}                                   # Initialize an empty dictionary to store word scores.

        if method == 'mi':
            mi = mutual_info_classif(X, y, discrete_features = True) # Measures how much information a word provides about the class.
            word_scores = dict(zip(feature_names, mi))               
        elif method == 'chi2':
            chi2_scores, _ = chi2(X, y) # Evaluate the independence between word and class (higher = more dependent = more important).
            word_scores = dict(zip(feature_names, chi2_scores))
        elif method == 'nmi':
            word_scores = { # Calculates normalized mutual information between each column (word) and the classes.
                word: normalized_mutual_info_score(X[:,i].toarray().flatten(), y)
                for i, word in enumerate(feature_names)
            }
        else:
            raise ValueError(f"Método no soportado: {method}")

        top_k_words = dict(sorted(word_scores.items(), key = lambda x: abs(x[1]), reverse = True)[:k])
        self.create_wordcloud(k, top_k_words, f"Top {k} informational words for {label_type}")

    def plot_class_representative_wordcloud(self, class_label: str, label_type: str, k: int = 50):
        """
        Generates a word cloud for a specific class label, highlighting words that best represent that class.

        Parameters
        ----------
        class_label : str
            Class label to analyze. Options are '1', '2', '3', etc. (for Polarity), specific town names (for Town)
            or specific types (for Type).
        label_type : str
            Type of label to use for feature extraction. Options are:
            - 'Polarity'
            - 'Town'
            - 'Type'
        k : int
            Number of top words to include in the word cloud.
        """
        if label_type == 'Polarity':
            y = [int(p[0]) for p in self.labels]
        elif label_type == 'Town':
            y = [p[1] for p in self.labels]
        elif label_type == 'Type':
            y = [p[2] for p in self.labels]
        else:
            raise ValueError("label_type must be 'Polarity', 'Town' or 'Type'")
        
        if class_label not in set(y):
            raise ValueError(f"'{class_label}' is not a valid label for {label_type}.")

        vectorizer = CountVectorizer(max_features = 5000)  # Create a CountVectorizer instance to convert text to a matrix of token counts.
        X = vectorizer.fit_transform(self.corpus)          # Transform the corpus into a bag-of-words representation (n_documents x n_features).
        feature_names = vectorizer.get_feature_names_out() # Contains the words mapping to the columns of the matrix.
        
        # Tokenize the corpus and keep only words in the vocabulary. This ignores the weird words or gramatical errors.
        vocab_set = set(feature_names)
        tokenized_texts = [ 
            [word for word in doc.split() if word in vocab_set] for doc in self.corpus
        ]

        alpha = 0.01
        target_counter = Counter() # Count words in the target class.
        other_counter = Counter()  # Count words in the other classes.

        # Count the occurrences of each word in the texts for the target class and other classes.
        for text, label in zip(tokenized_texts, y):
            if label == class_label:
                target_counter.update(text) # Count words in the target class.
            else:
                other_counter.update(text)  # Count words in the other classes.

        # Calculate the log odds ratio for each word in the vocabulary.
        vocab = set(target_counter.keys()) | set(other_counter.keys()) # Union of both counters.
        top_words = {}
        for word in vocab:
            a = target_counter[word] + alpha                      # Frequency of the word in the target class + alpha.
            b = other_counter[word] + alpha                       # Frequency of the word in the other classes + alpha.
            A = sum(target_counter.values()) + alpha * len(vocab) # Total frequency of words in the target class.
            B = sum(other_counter.values()) + alpha * len(vocab)  # Total frequency of words in the other classes.
            log_odds = np.log(a / (A - a)) - np.log(b / (B - b))  # Log odds ratio.
            top_words[word] = log_odds

        top_k_words = dict(sorted(top_words.items(), key = lambda x: x[1], reverse = True)[:k])
        self.create_wordcloud(k, top_k_words, f"Top {k} informational words for {NameFormatter.name_formatter(class_label)} in {label_type}")

    @staticmethod
    def create_wordcloud(max_words: int, dict_freq: dict, title: str):
        wordcloud = WordCloud(
            width = 1200, height = 600, background_color = 'black', colormap = 'winter',
            max_words = max_words
        ).generate_from_frequencies(dict_freq)

        plt.figure(figsize = (14,7), facecolor = 'black')
        plt.imshow(wordcloud, interpolation = 'bilinear')
        plt.axis("off")
        plt.title(title, fontsize = 20, color = 'white')
        plt.tight_layout(pad = 0)
        plt.show()

class MapPlotter:
    """
    MapPlotter class for plotting maps with regions and towns. This class takes a GeoDataFrame as input
    and provides methods to plot the map with regions and towns.
    """
    def __init__(self,  df: pd.DataFrame, gdf:gpd.GeoDataFrame):
        self.gdf = gdf
        self.data = df
        
        self.data["Region"] = self.data["Region"].apply(lambda x: self.stateFormatter(x))
        self.gdf["NOMGEO"] = self.gdf["NOMGEO"].apply(lambda x: self.stateFormatter(x))
        

    def stateFormatter(self, state: str) -> str:
        state = state.lower()
        state = unicodedata.normalize("NFD", state)
        state = re.sub(r'[\u0300-\u036f]', '', state)
        state = re.sub(r'\s+|_', '', state)
        state = re.sub(r'[^\w]', '', state)
        
        if state == "mexico":
            state = "estadodemexico"
        if state == "michoacandeocampo":
            state = "michoacan"
        if state == "veracruzdeignaciodelallave":
            state = "veracruz"
        if state == "coahuiladezaragoza":
            state = "coahuila"
        return state        

    def merge_states(self) -> pd.DataFrame:
        """
        Merges the GeoDataFrame with the DataFrame containing regions and towns.
        """
        data = self.gdf.copy()
        data['NOMGEO'] = data['NOMGEO'].apply(lambda x: self.stateFormatter(x))
        data = data.rename(columns = {'NOMGEO': 'Region'})
        
        number_of_reviews = self.data.groupby('Region').size().reset_index(name='counts')
        
        data['counts'] = data['Region'].map(
            number_of_reviews.set_index('Region')['counts']
        )
        
        return data
        
    def plot_map(self, ax = None, 
                 params = None, 
                 params_bar=None) -> None:

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
        
        
        if ax is None:
            fig, ax = plt.subplots(figsize = (20,6))
        
        data.plot(ax = ax, **params)
        cbar = ax.get_figure().get_axes()[1]
        cbar.set_title(**params_bar)
        ax.set_axis_off()
        plt.tight_layout(pad = 0)
        plt.show()
        
        