""" Now we need to make a train loop """


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
from optimizer import ScheduleOptim
from bert_database import BERTDataset
from model import BERT, BERTLM


class BERTTrainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 test_dataloader=None,
                 lr=1e-4,
                 weight_decay = 0.01,
                 betas = (0.9, 0.999),
                 warmup_steps = 10000,
                 log_freq = 10,
                 device = 'cuda'):
        self.device = device
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.model = model


        # Setting the adam optimizer and hyper-param
        self.optim = Adam(self.model.parameters(), lr = lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduleOptim(optimizer=self.optim, d_model= self.model.bert.d_model, n_warmup_steps= warmup_steps)

        # Using the negative loglikelihood Loss function for predicting the masked token
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        print("Total parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def iteration(self, data_loader, epoch, train=True):
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        mode = 'train' if train else 'test'

        # progess bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total = len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )        

        for i, data in data_iter:

            # 0. Batch data will be sent to device, CPU or GPU
            data = {key: value.to(self.device) for key, value in data.items()}
            
            # 1. forward the nsp and masked lm 
            next_sen_output, mask_lm_output = self.model.forward(data['bert_input'], data['segment_label'])

            # 2-1 NLL (negative log likelihood) loss of is_next classification
            next_loss = self.criterion(next_sen_output, data['is_next'])

            # 2-2 NLL of predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1,2) , data["bert-label"])

            loss = next_loss + mask_loss

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
            
            # 3. next sentence prediction accuracy
            correct = next_sen_output.argmax(dim=-1).eq(data['is_next']).sum().item()
            avg_loss += loss
            total_correct += correct
            total_element += data['is_next'].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i+1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        print(
            f"EP {epoch}, {mode}: \
                avg_loss = {avg_loss / len(data_iter)} \
                total_acc = {total_correct * 100.0 / total_element}"
        )
    

    def train(self, epoch):
        self.iteration(self.train_data, epoch)
    def test(self, epoch):
        self.iteration(self.test_data, epoch, train=False)


def main():
    ''' run the program...'''

    # lets initiate the saved tokenizer
    tokenizer = BertTokenizer.from_pretrained('./bert-it-l/bert-it-vocab.txt', local_files_only=True)

    train_data = BERTDataset(data_pair=pairs, seq_len= MAX_LEN, tokenizer=tokenizer)
    print(train_data)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)

    bert_model = BERT(
        vocab_size= len(tokenizer.vocab),
        d_model= 768,
        n_layers=2,
        heads = 12,
        dropout=0.1
    )

    bert_lm = BERTLM(bert=bert_model, vocab_size= len(tokenizer.vocab))
    bert_trainer = BERTTrainer(bert_lm, train_loader, device='cpu')
    epochs = 20

    for epoch in range(epochs):
        bert_trainer.train(epoch=epoch)

if __name__ == '__main__':
    main()
