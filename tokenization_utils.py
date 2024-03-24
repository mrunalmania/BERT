# here we use the BERT wordpeice tokenizer

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
from movie_datasets import pairs

# we need to save the data as the text file.
os.mkdir('./data')
text_data = []
file_count = 0

for sample in tqdm.tqdm(x[0] for x in pairs):
    text_data.append(sample)

    # once we hit the 10k mark then we save to file
    if len(text_data) == 10000:
        with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_data))
        text_data = []
        file_count+=1
paths = [str(x) for x in Path('./data').glob('**/*.txt')]

# training tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text = True,
    handle_chinese_chars = True,
    strip_accents = False,
    lowercase = True
)

tokenizer.train(
    files = paths,
    vocab_size = 30_000,
    min_frequency = 5,
    limit_alphabet = 1000,
    wordpieces_prefix = '##',
    special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
)

os.mkdir('./bert-it-l')
tokenizer.save_model('./bert-it-l', 'bert-it')
tokenizer = BertTokenizer.from_pretrained('./bert-it-l/bert-it-vocab.txt', local_files_only=True)