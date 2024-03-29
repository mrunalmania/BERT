# here we need to deifne the dataset for bert model to use.

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



class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len=64) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair
    
    def __len__(self):
        return self.corpus_lines
    

    def get_corpus_line(self, item):
        '''return the sentence pair'''
        return self.lines[item][0], self.lines[item][1]
    
    def get_random_line(self):
        '''return the random line from the corpus (single sentence)'''
        return self.lines[random.randrange(len(self.lines))][1]
    

    # we first need to define the random_word function
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []


        # 15% of the tokens would be replaced.
        # we use the random.random() method, which gives us the floating point number between 0,1.
        for i, token in enumerate(tokens):
            prob = random.random()

            # we need to remobe the [CLS] and [SEP] token which append by tokenizer
            token_id = self.tokenizer(token)['input_ids'][1:-1]

            if prob < 0.15:
                prob/=0.15

                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))
                
                else:
                    for i in range(len(token_id)):
                        output.append(token_id)
                output_label.append(token_id)
            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)
            
            # now we need to do the flattening
            output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
            output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
            assert len(output) == len(output_label)
            return output, output_label

    def get_sent(self, index):
        ''' return the random sentence pair'''
        t1, t2 = self.get_corpus_line(index)

        # negative or positive pair, for NSP
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0


    def __getitem__(self, item):

        # step 1: get the random sentence pair:
        t1, t2, is_next_label = self.get_sent(item)

        # Step 2: replace the random words in sentence with mask/ random words
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # step 3: adding the [cls] and [sep] token
        t1 = [self.tokenizer.vocab['[CLS]']] + t1_random + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_random + [self.tokenizer.vocab['[SEP]']]

        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        # step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len

        segment_label = [1 for _ in range(len(t1))] + [2 for _ in range(len(t2))][:self.seq_len]
        bert_input = (t1+t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]

        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert-label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}
        return {key: torch.tensor(value) for key, value in output.items()}



tokenizer = BertTokenizer.from_pretrained('./bert-it-l/bert-it-vocab.txt', local_files_only=True)
train_data = BERTDataset(pairs, seq_len=MAX_LEN, tokenizer=tokenizer)
print(train_data[random.randrange(len(train_data))])