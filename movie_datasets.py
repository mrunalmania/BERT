import os
from pathlib import Path
import torch
import re
import random
import transformers, datasets

MAX_LEN = 64

corpus_movi_conv = '.\datasets\cornell movie-dialogs corpus\movie_conversations.txt'
corpus_movi_lines = '.\datasets\cornell movie-dialogs corpus\movie_lines.txt'

with open(corpus_movi_conv, 'r', encoding='iso-8859-1') as c:
    conv = c.readlines()
with open(corpus_movi_lines, 'r', encoding='iso-8859-1') as l:
    lines = l.readlines()

# splitting the text using special token
lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]

# generate the question answer pairs
pairs = []
for con in conv:
    ids = eval(con.split(' +++$+++ ')[-1])
    for i in range(len(ids)):
        qa_pairs = []
        if i == len(ids)-1:
            break
        first = lines_dic[ids[i]].strip()
        second = lines_dic[ids[i+1]].strip()
        qa_pairs.append(" ".join(first.split()[:MAX_LEN]))
        qa_pairs.append(" ".join(second.split()[:MAX_LEN]))
        pairs.append(qa_pairs)
    

print(pairs[20])