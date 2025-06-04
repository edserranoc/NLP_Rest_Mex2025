import numpy as np
import pandas as pd
from typing import List, Tuple, Callable
import random

import torch
from nltk.tokenize import TweetTokenizer



class TextPreprocessor:
    """
    TextPreprocessor class for preprocessing text data. This class takes a tokenizer and a set of stopwords
    as input and provides a method to process text data. The text data is tokenized, converted to lowercase,
    and stopwords are removed. The processed text data is returned as a list of tokens.
    """
    def __init__(self, tokenizer: Callable = None, stopwords: set = None):
        """
        Initialize the TextPreprocessor with a tokenizer and a set of stopwords.
        
        Parameters
        ----------
        tokenizer : Callable, optional
            Tokenizer function to use for tokenizing the text data. If None, the default tokenizer is used.
        stopwords : set, optional
            Set of stopwords to remove from the text data. If None, no stopwords are removed.
        """
        self.tokenizer = tokenizer or TweetTokenizer().tokenize
        self.stopwords = stopwords or set()

    def process(self, text: str) -> List[str]:
        """
        Process the text data by tokenizing, converting to lowercase, and removing stopwords.

        Parameters
        ----------
        text : str
            Text data to process.

        Returns
        -------
        tokens : List[str]
            List of tokens extracted from the text data. The tokens are in lowercase and stopwords are removed.
        """
        tokens = self.tokenizer(text.lower())
        return [t for t in tokens if t.isalpha()]


class EmbeddingLoader:
    """
    EmbeddingLoader class for loading word embeddings from a file. This class takes the path to the embeddings
    file as input and provides a method to load the embeddings. The embeddings are loaded into a vocabulary
    dictionary and an embedding matrix. The vocabulary dictionary maps words to their corresponding indices
    in the embedding matrix. The embedding matrix contains the word embeddings for each word in the vocabulary.
    """
    def __init__(self, embeddings_path: str):
        """
        Initialize the EmbeddingLoader with the path to the embeddings file.

        Parameters
        ----------
        embeddings_path : str
            Path to the file containing the word embeddings. The file should be in the format:
                word1 embedding1
                word2 embedding2
                ...
        """
        self.embeddings_path = embeddings_path

    def load(self, device: str) -> Tuple[dict, torch.Tensor]:
        """
        Load the word embeddings from the file and return the vocabulary and embedding matrix.

        Returns
        -------
        vocab : dict
            Dictionary mapping words to their corresponding indices in the embedding matrix.
        emb_mat : torch.Tensor
            Embedding matrix containing the word embeddings for each word in the vocabulary.
        """
        vocab = {}
        embeddings_list = []
        with open(self.embeddings_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                values = line.split()
                vocab[values[0]] = i + 1
                embeddings_list.append(np.asarray(values[1:], dtype = 'float32'))
        embeddings_list.insert(0, np.mean(np.vstack(embeddings_list), axis = 0)) # Mean vector for unk
        embeddings_list.insert(0, np.zeros(100))                                 # Padding vector
        vocab['pad'] = 0
        vocab['unk'] = 1
        emb_mat = torch.tensor(np.vstack(embeddings_list), dtype = torch.float32, device = device)

        return vocab, emb_mat
    
    
def augment_input_ids(input_ids: torch.Tensor, length: int, pad_token: int = 0) -> torch.Tensor:
    n_tokens = input_ids.size(0)
    padding_amount = n_tokens - length

    # Solo aplica si hay al menos 20% de padding
    if padding_amount < int(0.2 * n_tokens) or length <= 5:
        return input_ids

    # Elige posición aleatoria para extraer subsecuencia de tamaño 5
    start_idx = random.randint(0, max(0, length - 6))
    subseq = input_ids[start_idx:start_idx + 5]

    # Inserta la subsecuencia justo después del final real
    new_ids = input_ids.clone()
    insert_pos = length
    end_pos = min(insert_pos + 5, n_tokens)
    new_ids[insert_pos:end_pos] = subseq[:end_pos - insert_pos]

    return new_ids