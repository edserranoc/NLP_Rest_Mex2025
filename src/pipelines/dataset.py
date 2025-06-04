# ReviewDataset, ReviewDatasetTrain, DatasetSplitter, collate_fn_hierarchical

from typing import List, Tuple
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from src.pipelines.preprocessor import TextPreprocessor, augment_input_ids



class ReviewDataset(Dataset):
    """
    ReviewDataset class for creating a dataset from corpus and labels. This class takes a list of corpus,
    a list of labels, a vocabulary dictionary, and a text preprocessor as input. The corpus are tokenized
    and converted to indices using the vocabulary dictionary. The labels are returned as tensors. The dataset
    can be used with a DataLoader for batching and shuffling.
    """
    def __init__(self, corpus: List[str], labels: List[Tuple[int, str, str]],
                 vocab: dict, device: str, processor: TextPreprocessor, max_len: int = 10000):
        """
        Initialize the ReviewDataset with corpus, labels, vocabulary, and text preprocessor.
        """
        self.corpus = corpus
        self.labels = labels
        self.vocab = vocab
        self.processor = processor
        self.max_len = max_len
        self.device = device

    def __len__(self) -> int:
        """
        Return the number of corpus in the dataset.
        """
        return len(self.corpus)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, torch.Tensor, str, str]:
        tokens = self.processor.process(self.corpus[idx])
        ids = [self.vocab.get(t, self.vocab['unk']) for t in tokens[:self.max_len]]
        length = len(ids)
        ids += [self.vocab['pad']] * (self.max_len - length)
        polarity, town, place_type = self.labels[idx]
        return (
            torch.tensor(ids, dtype = torch.long, device = self.device),
            length,
            torch.tensor(polarity, dtype = torch.float, device = self.device),
            town,
            place_type
        )
        

class ReviewDatasetTrain(Dataset):
    def __init__(self, corpus, labels, vocab, processor, max_len, device,
                 rare_towns: set, rare_polarities: set = {1, 2, 3}):
        self.corpus = corpus
        self.labels = labels
        self.vocab = vocab
        self.processor = processor
        self.max_len = max_len
        self.device = device
        self.rare_towns = rare_towns
        self.rare_polarities = rare_polarities

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        tokens = self.processor.process(self.corpus[idx])
        ids = [self.vocab.get(t, self.vocab['unk']) for t in tokens[:self.max_len]]
        length = len(ids)
        ids += [self.vocab['pad']] * (self.max_len - length)

        polarity, town, place_type = self.labels[idx]
        ids_tensor = torch.tensor(ids, dtype=torch.long)

        if polarity in self.rare_polarities or town in self.rare_towns:
            ids_tensor = augment_input_ids(ids_tensor, length, pad_token=self.vocab['pad'])

        return (
            ids_tensor.to(self.device),
            length,
            torch.tensor(polarity, dtype=torch.float, device=self.device),
            town,
            place_type
        )
        
        
class DatasetSplitter:
    def __init__(self, reviews: List[str], labels: List[Tuple[int, str, str]]):
        self.reviews = reviews
        self.labels = labels

        # Estratificación múltiple usando Polaridad y Pueblo
        self.strat_labels = LabelEncoder().fit_transform([
            f"{polarity}_{town}" for polarity, town, _ in labels
        ])

    def stratified_split(
            self, vocab: dict, processor, device: str, test_size: float = 0.2, random_state: int = 42,
            max_len: int = 10000) -> Tuple['ReviewDatasetTrain', 'ReviewDataset']:
    
        splitter = StratifiedShuffleSplit(n_splits = 1, test_size = test_size, random_state = random_state)
        for train_idx, val_idx in splitter.split(self.reviews, self.strat_labels):
            train_reviews = [self.reviews[i] for i in train_idx]
            train_labels = [self.labels[i] for i in train_idx]

            val_reviews = [self.reviews[i] for i in val_idx]
            val_labels = [self.labels[i] for i in val_idx]

#            train_dataset = ReviewDataset(corpus = train_reviews, labels = train_labels, vocab = vocab,
#                                          processor = processor, max_len = max_len, device = device)

            rare_towns = {town for town, count in Counter(t for _, t, _ in train_labels).items() if count < 2800}
            train_dataset = ReviewDatasetTrain(corpus = train_reviews, labels = train_labels, vocab = vocab,
                                               processor = processor, max_len = max_len, device = device,
                                               rare_towns = rare_towns)
            val_dataset = ReviewDataset(corpus = val_reviews, labels = val_labels, vocab = vocab,
                                        processor = processor, max_len = max_len, device = device)

            return train_dataset, val_dataset
        

def collate_fn_hierarchical(batch: List[Tuple[torch.Tensor, int, float, str]],device: str = None
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str], torch.Tensor, torch.Tensor]:
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    grouped = defaultdict(list)
    for input_ids, length, polarity, town, place_type in batch:
        grouped[town].append((input_ids, length, polarity, place_type))

    towns = list(grouped.keys())
    max_reviews = max(len(reviews) for reviews in grouped.values())
    max_len = max(len(ids) for reviews in grouped.values() for ids, _, _, _ in reviews)

    batch_input_ids  = []
    batch_lengths    = []
    batch_polarities = []
    batch_types      = []
    review_masks     = [] # 1 si es una reseña válida, 0 si es dummy
    token_masks      = [] # 1 si es un token válido, 0 si es pad

    for town in towns:
        reviews     = grouped[town]
        padded_input_ids = []
        lengths     = []
        polarities  = []
        types       = []
        review_mask = []
        token_mask  = []

        for ids, length, polarity, place_type in reviews:
            pad_len = max_len - len(ids)
            padded = torch.cat([
                ids,
                torch.zeros(pad_len, dtype = torch.long, device = device)
            ])
            mask = torch.cat([
                torch.ones(len(ids), device = device),
                torch.zeros(pad_len, device = device)
            ])

            padded_input_ids.append(padded)
            token_mask.append(mask)
            lengths.append(length)
            polarities.append(polarity.to(device))
            types.append(place_type)
            review_mask.append(1)

        while len(padded_input_ids) < max_reviews:
            padded_input_ids.append(torch.zeros(max_len, dtype = torch.long, device = device))
            token_mask.append(torch.zeros(max_len, device = device))
            lengths.append(0)
            polarities.append(torch.tensor(0., device = device))
            types.append("None")
            review_mask.append(0)

        batch_input_ids.append(torch.stack(padded_input_ids))        # (n_reviews, n_tokens)
        batch_lengths.append(torch.tensor(lengths, device = device)) # (n_reviews,)
        batch_polarities.append(torch.stack(polarities))             # (n_reviews,)
        batch_types.append(types)
        review_masks.append(torch.tensor(review_mask, device = device)) # (n_reviews,)
        token_masks.append(torch.stack(token_mask))                     # (n_reviews, n_tokens)

    input_ids = torch.stack(batch_input_ids).to(device) # (batch_size, n_reviews, n_tokens)
    lengths = torch.stack(batch_lengths).to(device)
    polarities = torch.stack(batch_polarities).to(device)
    review_masks = torch.stack(review_masks).to(device)
    token_masks = torch.stack(token_masks).to(device)

    return input_ids, lengths, polarities, towns, batch_types, review_masks, token_masks