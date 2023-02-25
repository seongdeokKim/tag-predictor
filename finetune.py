import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

from helper.models.bert_labeler import BertLabeler
from helper.trainer import Trainer
from helper.data_loader import CustomDataset, TokenizerWrapper
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np
from scipy import sparse

# Block the warning message of tokenizer
import logging
logging.disable(logging.WARNING)

def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', default='/home/workspace/src/tag_predictor/helper/saved_model.pth')

    p.add_argument('--train_fn', default='/home/workspace/src/tag_predictor/preprocessed_train.csv')

    p.add_argument('--tag_index_path', type=str, default='/home/workspace/src/tag_predictor/top_6000_tag_index.txt')
    p.add_argument('--pretrain_path', type=str, default='bert-base-uncased')
    p.add_argument('--checkpoint_path', type=str, default=None)
    # p.add_argument('--checkpoint_path', type=str, default='/home/workspace/src/tag_predictor/helper/saved_model.pth')

    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])

    p.add_argument('--batch_size', type=int, default=400)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--warmup_ratio', type=float, default=.1)
    p.add_argument('--beta1', type=float, default=.9)
    p.add_argument('--beta2', type=float, default=.999)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', action='store_true')

    p.add_argument('--max_length', type=int, default=200)
    p.add_argument('--classifier_dropout', type=float, default=.1)

    config = p.parse_args()

    return config

def get_tag_index_map(tag_index_path):
    index_to_tag = {}
    tag_to_index = {}
    with open(tag_index_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            if line.strip() != '':
                field_line = line.strip().split('\t')
                idx = field_line[0].strip()
                tag = field_line[1].strip()

                index_to_tag[int(idx)] = str(tag)
                tag_to_index[str(tag)] = int(idx)

    return index_to_tag, tag_to_index

def read_csv_file(input_path, index_to_tag, tag_to_index):
    df = pd.read_csv(input_path)
    df = df.iloc[:400000, :]

    questions = df["question"].tolist()

    multilabels = np.zeros((len(questions), len(tag_to_index.keys())))
    multilabels = multilabels.T

    vectorizer = CountVectorizer(tokenizer= lambda text : text.split(), binary=True)
    src_multilabels = vectorizer.fit_transform(df["tags"]).toarray()
    src_multilabels = src_multilabels.T
    src_tag_list = list(vectorizer.get_feature_names_out())
    src_tag_set = set(src_tag_list)

    overlap_tags = []
    for tag in tag_to_index.keys():
        if tag in src_tag_set:
            overlap_tags.append(tag)

    for tag in overlap_tags:
        multilabels[tag_to_index[tag]] = src_multilabels[src_tag_list.index(tag)]

    multilabels_df = pd.DataFrame(multilabels.T, columns=tag_to_index.keys())

    return (questions, multilabels_df)

def get_loader(config, dataset_path, tokenizer, index_to_tag, tag_to_index):

    (questions, multilabels_df) = read_csv_file(dataset_path, index_to_tag, tag_to_index)
    
    train_idx = int(len(questions)*.7)
    valid_idx = int(len(questions)*.85)

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        CustomDataset(questions[:train_idx], 
                      multilabels_df.iloc[:train_idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TokenizerWrapper(tokenizer,
                                    config.max_length).collate,
    )
    valid_loader = DataLoader(
        CustomDataset(questions[train_idx:valid_idx], 
                      multilabels_df.iloc[train_idx:valid_idx]),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=TokenizerWrapper(tokenizer,
                                    config.max_length).collate,
    )
    test_loader = DataLoader(
        CustomDataset(questions[valid_idx:], 
                      multilabels_df.iloc[valid_idx:]),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=TokenizerWrapper(tokenizer,
                                    config.max_length).collate,
    )

    return train_loader, valid_loader, test_loader

def main(config):

    # Get index_to_tag
    index_to_tag, tag_to_index = get_tag_index_map(config.tag_index_path)

    # Get pretrained tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.pretrain_path)

    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, test_loader = get_loader(config, config.train_fn, 
                                                         tokenizer, 
                                                         index_to_tag, tag_to_index)

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
        '|test| =', len(test_loader) * config.batch_size,
    )

    loss_func = nn.CrossEntropyLoss(reduction='sum')

    model = BertLabeler(
        config,
        num_tags=len(index_to_tag),
        pretrain_path=config.pretrain_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    if torch.cuda.is_available() and config.device == "cuda":
        print("Using", torch.cuda.device_count(), "GPUs!")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        loss_func = loss_func.to(device)

        if config.checkpoint_path is not None:
            checkpoint = torch.load(config.checkpoint_path, 
                                    map_location='cuda')['bert']
            msg = model.load_state_dict(checkpoint, strict=True)

    else:
        model = model.to(device)

        if config.checkpoint_path is not None:
            checkpoint = torch.load(config.checkpoint_path, 
                                    map_location='cpu')['bert']
            msg = model.load_state_dict(checkpoint, strict=True)

    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # optimizer = optim.Adam(model.parameters())

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters,
                                lr=config.lr,
                                betas=(config.beta1, config.beta2),
                                eps=config.adam_epsilon)

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                n_warmup_steps,
                                                n_total_iterations)

    # Start train.
    trainer = Trainer(config)

    best_model = trainer.train(model,
                               loss_func,
                               optimizer,
                               scheduler,
                               train_loader,
                               valid_loader,
                               index_to_tag,
                               device)

    trainer.test(best_model,
                 test_loader,
                 index_to_tag,
                 device)
        
    torch.save({
        'bert': best_model.state_dict(),
        'config': config,
        'pretrain_path': config.pretrain_path,
        'vocab': tokenizer.get_vocab(),
        'index_to_tag': index_to_tag,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]= "2,3,4,5"

    config = define_argparser()
    main(config)
