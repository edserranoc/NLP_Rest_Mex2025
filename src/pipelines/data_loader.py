import pandas as pd
from typing import List, Tuple


class DataLoader2:
    """
    DataLoader class for loading and processing data from a CSV file. This class reads the data from a CSV
    file, adds a sentiment column based on the polarity, and provides methods to extract reviews and labels.
    """
    def __init__(self, data_path: str):
        """
        Initialize the DataLoader with the path to the data file.

        Parameters
        ----------
        data_path : str
            Path to the CSV file containing the data.
        """
        self.data_path = data_path
        self.data = self._read_data()
        self._add_sentiment_column()

    def _read_data(self) -> pd.DataFrame:
        """
        Read the data from the CSV file.
        """
        return pd.read_csv(self.data_path)

    def _add_sentiment_column(self):
        """
        Add a sentiment column to the DataFrame based on the polarity.
        """
        self.data['Sentiment'] = self.data['Polarity'].apply(
            lambda x: 'Positive' if x >= 4 else 'Neutral' if x == 3 else 'Negative'
        )

    def get_data(self) -> pd.DataFrame:
        """Return a copy of the DataFrame."""
        return self.data.copy()
    
class ReviewLabelExtractor:
    """
    ReviewLabelExtractor class for extracting reviews and labels from a DataFrame. This class takes a DataFrame
    as input and provides a method to extract reviews and labels. The reviews are extracted from the 'Review'
    column and the labels are extracted from the 'Polarity', 'Town', and 'Type' columns. The labels are returned
    as a list of tuples.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ReviewLabelExtractor with a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data. The DataFrame should contain the columns 
                'Review', 'Polarity', 'Town', 'Type'.
        """
        self.df = df

    def extract(self) -> Tuple[List[str], List[Tuple[int, str, str]]]:
        """
        Extract reviews and labels from the DataFrame, cleaning missing values and normalizing text.
    
        Returns
        -------
        reviews : List[str]
            List of reviews extracted from the 'Review' column of the DataFrame.
        labels : List[Tuple[int, str, str]]
            List of tuples containing the polarity, town, and type extracted from the DataFrame.
            Format: (polarity, town, type)
        """
        # Eliminar filas con etiquetas faltantes
        self.df.dropna(subset = ['Polarity', 'Town', 'Type'], inplace = True)
    
        # Normalizar etiquetas
        self.df['Polarity'] = self.df['Polarity'].astype(int)
        self.df['Town'] = self.df['Town'].str.strip()
        self.df['Type'] = self.df['Type'].str.strip()
        self.df['Title'] = self.df['Title'].fillna('')
    
        labels = list(zip(
            self.df['Polarity'],
            self.df['Town'],
            self.df['Type']
        ))

        # Extraer datos
        self.df['Review'] = self.df['Title'] + ' ' + self.df['Review']
        self.df.drop(columns=['Title'], inplace=True)

        self.df['Review'] = self.df['Review'].str.lower()

        fix_map = {
            'ãº': 'ú',
            'ã¡': 'á',
            'ã!': 'ó',
            'ã©': 'é',
            'ã³': 'ó',
            'ã²': 'ó',
            'ã±': 'ñ',
            'â¡': '¡',
            'å': 'a',
            'ê': 'e',
            'ã': 'í',
            '½': 'medio',
            '¼': 'cuarto',
            ':))': ' feliz ',
            ':)': ' feliz ',
            ':(': ' triste ',
            '>:(': ' mal ',
            ' :/': ' confuso ',
            ':-)': ' feliz ',
            '\x85':''
        }
        symbols_to_remove = r'[,!¡¿\?%\.\(\)/\\_\-¢£¦¤¨©ª«¬®°²¹º»×æ÷·&"\'~*\{\}\|\+\@\[\\\]\=\`<>\:\;]'


        def fix_encoding_issues(text):
            for bad in fix_map:
                text = text.replace(bad, fix_map[bad])
            return text

        self.df["Review"] = self.df["Review"].apply(fix_encoding_issues)
        self.df["Review"] = self.df["Review"].str.replace(symbols_to_remove, '', regex=True)
        
        return self.df["Review"], labels