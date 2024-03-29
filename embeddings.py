'''
        Here we need to add the positional embeddings in our BERT model.
'''

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


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, d_model, max_len=128) -> None:
        super().__init__()

        # positional encoding (pe) in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        for pos in range(max_len):
            for i in range(0,d_model, 2):
                pe[pos, i] = math.sin(pos / (10000**((2*i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*(i+1))/d_model)))
        
        # include the batch size
        self.pe = pe.unsqueeze(0)
    
    def forward(self,x):
        return self.pe

class BERTEmbedding(torch.nn.Module):
    '''
        BERT Embedding is consisted with under feature:
        1. Token Embedding: normal embedding matrix
        2. Positional Embedding: adding positional information using sin, cos
        3. Segment embedding: adding the sentence segment info (sen_A:1 , sen_B: 2)
        sum of all these features are the output of the BERT Embedding
    '''

    def __init__(self, vocab_size, embed_size, seq_len = 64, dropout=0.1) -> None:
        super().__init__()
        self.embed_size = embed_size

        # 1. token embedding
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = torch.nn.Embedding(3, embedding_dim=embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)
    
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)





