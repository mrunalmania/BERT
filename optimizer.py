
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


class ScheduleOptim():
    ''' A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps) -> None:
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        """ Step with the inner optimization """
        self._update_learning_rate()
        self._optimizer.step()
    
    def zero_grad(self):
        self._optimizer.zero_grad()
    
    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5)
        ])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step'''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
