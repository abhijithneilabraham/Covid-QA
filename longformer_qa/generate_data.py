#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:29:06 2021

@author: abhijithneilabraham
"""


from datasets import load_dataset
import pandas as pd
def deepset_save():
    train = load_dataset("covid_qa_deepset",split="train")
    cols=["question","context","id","answers"]
    data=[]
    for question,context,document_id,answer in zip(train["question"],train["context"],train["document_id"],train["answers"]):
        data.append([question,context,document_id,answer])
    df=pd.DataFrame(data,columns=cols)
    df.to_json("train_deepset.jsonl",lines=True,orient='records')

