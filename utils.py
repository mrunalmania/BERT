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


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout=0.1) -> None:
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)

    
    def forward(self, query, key, value, mask):
        """
        query, key, value of shape = (batch_size, max_len, d_model)
        mask of shape = (batch_size, 1, 1, max_words)

        """

        # (batch_size, max_len, d_model) -> (batch_size, max_len, h, d_k) -> (batch_size, h, max_len, d_k)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        

        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0,2,1,3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0,2,1,3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0,2,1,3)

        scores = torch.matmul(query, key.permute(0,1,3,2)) / math.sqrt(query.size(-1))

        # fill 0 with supersmall number, so that it won't effect the softmax result.
        scores = scores.masked_fill(mask==0, -1e9)

        # need to apply softmax on scores
        weights =  F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # now we need to multiply it with value, context = softmax(QK^T/sqrt(d))*V
        context = weights @ value

        # we need to make context direction same as the input (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads*self.d_k)

        return context


# now we need to create feed forward network class
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, middle_dim = 2048, dropout = 0.1 ) -> None:
        super(FeedForward, self).__init__()

        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.L1Loss(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.GELU()
    
    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


# Now we need to create the encoder layer
class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model=768, heads = 12, feed_forward_hidden = 768*4, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()
        
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads=heads, d_model=d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
    
    def forward(self, embeddings, mask):
        # embeddings = (batch_size, max_len, d_model)
        # mask = (batch_size, 1, 1, max_len)
        # result = (batch_size, max_len, d_model)

        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))

        # residual network
        interacted = self.layernorm(embeddings + interacted)

        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded
    

