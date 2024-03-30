"""
Here we are going to build our final BERT model.
"""

# lets first import the lib.
import os
from pathlib import Path
import torch
import re
import random
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from movie_datasets import pairs, MAX_LEN
from embeddings import BERTEmbedding
from utils import EncoderLayer

# now we are going to develop the BERT class
class BERT(torch.nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1) -> None:
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.n_layers = n_layers


        # as per the BERT paper: they have used 4*hidden_size for FFN
        self.feed_forward_hidden  = d_model * 4

        # embedding for BERT and sum of positional, segment, and token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model)

        # multi-layer transformer block
        self.encoded_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model*4, dropout=0.1) for _ in range(n_layers)]
        )

    
    def forward(self, x, segment_info):

        # attention masking for padded token
        mask = (x>0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed seq to seq of vectors
        x = self.embedding(x, segment_info)
        
        # running over multiple transformer
        for encoder in self.encoded_blocks:
            x = encoder.forward(x, mask)
        return x

class NextSentencePrediction(torch.nn.Module):
    def __init__(self, hidden) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(hidden, 2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:,0]))


class MaskedLanguageModel(torch.nn.Module):
    def __init__(self, hidden, vocab_size) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class BERTLM(torch.nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction + Masked Language Model
    """
    def __init__(self, bert:BERT, vocab_size:int) -> None:
        super().__init__()

        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size=vocab_size)
    
    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)
