from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Parameter(torch.randn(hidden_size))

    def forward(self, encoder_outputs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        u = torch.tanh(self.linear(encoder_outputs))  # (B, T, H)
        scores = torch.matmul(u, self.context_vector) # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(scores, dim = 1)     # (B, T)
        weighted_output = torch.sum(encoder_outputs * attn_weights.unsqueeze(-1), dim = 1)  # (B, H)
        return weighted_output, attn_weights
    
class HierarchicalMultiTaskModel(nn.Module):
    def __init__(self,  vocab_size: int, 
                        embed_dim : int,
                        hidden_size: int,
                        num_polarities: int = 5,
                        num_types: int = 3,
                        num_towns: int = 40,
                        embedding_matrix = None):
        
        super().__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze = False, padding_idx = 0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = 0)

        self.dropout_embed = nn.Dropout(0.15)
        self.dropout_head = nn.Dropout(0.15)

        self.word_rnn = nn.GRU(embed_dim, hidden_size, batch_first = True, bidirectional = True)
        self.word_attention = Attention(2 * hidden_size)

        self.sent_rnn = nn.GRU(hidden_size * 2, hidden_size, batch_first=True, bidirectional = True)
        self.sent_attention = Attention(2 * hidden_size)

        self.polarity_fc = nn.Linear(hidden_size * 2, num_polarities)
        self.type_fc = nn.Linear(hidden_size * 2, num_types)
        self.town_fc = nn.Linear(hidden_size * 2, num_towns)

    def forward(self, input_ids, lengths, review_mask):
        batch_size, n_reviews, n_tokens = input_ids.size()

        input_ids = input_ids.view(-1, n_tokens)
        lengths = lengths.view(-1)
        input_ids = input_ids.to(dtype = torch.long)
        valid_mask = lengths > 0

        valid_input_ids = input_ids[valid_mask]
        valid_lengths = lengths[valid_mask]

        embedded = self.embedding(valid_input_ids)
        embedded = self.dropout_embed(embedded)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        word_output, _ = self.word_rnn(packed)
        word_output, _ = nn.utils.rnn.pad_packed_sequence(word_output, batch_first=True)

        valid_input_ids = valid_input_ids[:, :word_output.size(1)]
        attn_mask = (valid_input_ids != 0)

        valid_review_repr, _ = self.word_attention(word_output, attn_mask)

        device = input_ids.device
        review_repr = torch.zeros((batch_size * n_reviews, valid_review_repr.shape[-1]), device=device)
        review_repr[valid_mask] = valid_review_repr
        review_repr = review_repr.view(batch_size, n_reviews, -1)

        sent_output, _ = self.sent_rnn(review_repr)
        sent_attn_mask = review_mask.bool()
        town_repr, _ = self.sent_attention(sent_output, sent_attn_mask)
        town_repr = self.dropout_head(town_repr)

        flat_sent_output = sent_output.reshape(-1, sent_output.shape[-1])
        flat_sent_output = self.dropout_head(flat_sent_output)

        polarity_logits = self.polarity_fc(flat_sent_output).view(batch_size, n_reviews, -1)
        type_logits = self.type_fc(flat_sent_output).view(batch_size, n_reviews, -1)
        town_logits = self.town_fc(town_repr)

        return polarity_logits, town_logits, type_logits