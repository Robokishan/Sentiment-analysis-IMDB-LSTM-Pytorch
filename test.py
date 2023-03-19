#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy

SEED = 1234
USE_GPU = False

torch.manual_seed(SEED)
tokenizer = get_tokenizer('spacy', language="en_core_web_sm")


device = torch.device('mps' if torch.backends.mps.is_available() and USE_GPU == True else 'cpu')
print(device)

train_iter, test_iter = IMDB(split=('train', 'test'))
train_list = list(train_iter)
test_list = list(test_iter)

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# unk_token = '<unk>'
pad_token = '<pad>'
special_tokens = [pad_token]

# Counter syntax and build_vocab_from_iterator is same . build_vocab_from_iterator uses counter inside the pytorch.
vocabulary = build_vocab_from_iterator(yield_tokens(train_iter), specials=special_tokens)
vocabulary.set_default_index(vocabulary[pad_token])

# unk_index = vocab[unk_token]
pad_index = vocabulary[pad_token]




new_stoi = vocabulary.get_stoi()
new_itos = vocabulary.get_itos()
# sentimentMap = {"pos": 0, "neg": 1}
# sentimentMap = {1: "neg", 2: "pos"}
sentimentMap = {1: 0, 2: 1}

text_transform = lambda x: vocabulary(tokenizer(x))

    
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)
    
# BUILDING MODEL
INPUT_DIM = len(vocabulary)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
BATCH_SIZE = 64
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = new_stoi[pad_token]

model = RNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)

def loadModel(model,filePath):    
    model.load_state_dict(torch.load(filePath, map_location=torch.device(device)))

loadModel(model, 'tut1-model.pt')

nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [new_stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

print(predict_sentiment(model, "This film is terrible"))
print(predict_sentiment(model, "This film is very good"))

