#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 23:08:02 2021

@author: abhijithneilabraham
"""

import pandas as pd
train=pd.read_json("train_deepset.jsonl",lines=True)
train_split=int(len(train)*0.9)
train_contexts, train_questions, train_answers = train["context"][:train_split].tolist(),train["question"][:train_split].tolist(),train["answers"][:train_split].tolist()
val_contexts, val_questions, val_answers =train["context"][train_split:].tolist(), train["question"][train_split:].tolist(),train["answers"][train_split:].tolist()

def add_end_idx(answers, contexts):
    a=[]
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        answer['answer_start']=answer['answer_start'][0]
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        answer['answer_end'] = end_idx
        a.append(answer)
    return a
train_answers=add_end_idx(train_answers, train_contexts)
val_answers=add_end_idx(val_answers, val_contexts)

from transformers import ElectraTokenizerFast
tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-base-discriminator')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    return encodings

train_encodings=add_token_positions(train_encodings, train_answers)
val_encodings=add_token_positions(val_encodings, val_answers)

import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

from transformers import ElectraForQuestionAnswering
model = ElectraForQuestionAnswering.from_pretrained("google/electra-base-discriminator")

from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
model.save_pretrained("covid_qa")

model.eval()
