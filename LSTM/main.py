#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab, build_vocab_from_iterator
from collections import Counter
import matplotlib.pyplot as plt
import tqdm
import time
from torchtext.vocab import GloVe, vocab
import spacy

# import random
# import functools

SEED = 1234
USE_GPU = True

torch.manual_seed(SEED)
tokenizer = get_tokenizer('spacy', language="en_core_web_sm")


device = torch.device('mps' if torch.backends.mps.is_available() and USE_GPU == True else 'cpu')
print(device)


train_iter, test_iter = IMDB(split=('train', 'test'))
train_list = list(train_iter)
test_list = list(test_iter)

print(f'Number of training examples: {len(train_list)}')
print(f'Number of validation examples: {len(test_list)}')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# unk_token = '<unk>'
pad_token = '<pad>'
special_tokens = [pad_token]

# Counter syntax and build_vocab_from_iterator is same . build_vocab_from_iterator uses counter inside the pytorch.

vocabulary = build_vocab_from_iterator(yield_tokens(train_iter), specials=special_tokens)

# counter = Counter()
# for (label, line) in train_iter:
#     counter.update(tokenizer(line))
# vocab = vocab(counter, min_freq=5, specials=special_tokens)


vocabulary.set_default_index(vocabulary[pad_token])

# unk_index = vocab[unk_token]
pad_index = vocabulary[pad_token]




new_stoi = vocabulary.get_stoi()
new_itos = vocabulary.get_itos()
# sentimentMap = {"pos": 0, "neg": 1}
# sentimentMap = {1: "neg", 2: "pos"}
sentimentMap = {1: 0, 2: 1}

text_transform = lambda x: vocabulary(tokenizer(x))


# Print out the output of text_transform
print("input to the text_transform:", "here is an example")
print("output of the text_transform:", text_transform("here is an example"))




# MODEL DEF

# class RNN(nn.Module):
#     def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(input_dim, embedding_dim)
#         self.rnn = nn.RNN(embedding_dim, hidden_dim)
#         self.fc = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, text):
#         embedded = self.embedding(text)
#         output, hidden = self.rnn(embedded)
#         # print("output --> ",output.shape, hidden.shape)
#         #output = [sent len, batch size, hid dim]
#         #hidden = [1, batch size, hid dim]
#         assert torch.equal(output[-1,:,:], hidden.squeeze(0))
#         return self.fc(hidden.squeeze(0))
    
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
print("dim", INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX, DROPOUT)

model = RNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

# https://github.com/pytorch/text/issues/1350#issuecomment-875807109
# https://pytorch.org/text/stable/vocab.html#torchtext.vocab.Vectors.get_vecs_by_tokens

MAX_VOCAB_SIZE = 25_000
myvec = GloVe(name="6B",dim=100)
pretrained_embedding = myvec.get_vecs_by_tokens(vocabulary.get_itos())

# https://github.com/pytorch/text/issues/1350#issuecomment-875807109
# https://pytorch.org/text/stable/vocab.html#torchtext.vocab.Vectors.get_vecs_by_tokens
print(pretrained_embedding.shape,myvec.vectors.shape )
with torch.no_grad():
    model.embedding.weight.copy_(pretrained_embedding)
glove_vectors = vocab(myvec.stoi)
my_embeddings = torch.nn.Embedding.from_pretrained(myvec.vectors,freeze=True) 

# add pretrained vocab vectors
len(vocabulary)

def collate_batch(batch):
    label_list, text_list, y_lens = [], [], []
    for index, (_label, _text) in enumerate(batch):
        label_list.append(sentimentMap[_label])
        _text_trans = text_transform(_text)
        processed_text = torch.tensor(_text_trans)
        text_list.append(processed_text)
        y_lens.append(len(_text_trans))
    y_lens = torch.tensor(y_lens)
    return {"labels": torch.tensor(label_list, dtype=torch.float32).to(device), "emb": pad_sequence(text_list, padding_value=0).to(device), "text_lengths": y_lens}


train_dataloader = torch.utils.data.DataLoader(list(train_iter), 
                                               batch_size=BATCH_SIZE, 
                                               collate_fn=collate_batch, 
                                               shuffle=True)

# valid_dataloader = torch.utils.data.DataLoader(list(valid_iter), batch_size=batch_size, collate_fn=collate)
test_dataloader = torch.utils.data.DataLoader(list(test_iter), batch_size=BATCH_SIZE, collate_fn=collate_batch)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

for index,i in enumerate(train_dataloader):
    labels = i["labels"]
    ids = i["emb"]
    lengths = i["text_lengths"]
    print(labels, ids)
    break

# optimizer = optim.SGD(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

# TRAINING AND VALIDATION
def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

all_preds = []
all_labels = []

def train(dataloader, model, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    t_batch = tqdm.tqdm(dataloader, desc='training...', file=sys.stdout)
    counter = 0
    for index,batch in enumerate(t_batch):
        label = batch["labels"]
        ids = batch["emb"]
        text_lengths = batch["text_lengths"]
        optimizer.zero_grad()
        
        predictions = model(ids,text_lengths)
        predictions = predictions.squeeze(1)
        all_preds.append(predictions)
        all_labels.append(label)

        loss = criterion(predictions, label)

        acc = binary_accuracy(predictions, label)
        
        loss.backward()
        
        optimizer.step()
        loss , accuracy = loss.item(), acc.item()
        epoch_loss += loss
        epoch_acc += accuracy
        counter += 1
        t_batch.set_description("loss %.2f accuracy %.2f Count: %.2f" % ((epoch_loss/(counter)), (epoch_acc/(counter)), counter))
        
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

# def evaluate(model, iterator, criterion):
#     epoch_loss = 0
#     epoch_acc = 0
#     model.eval()
    
#     with torch.no_grad():
#         for batch in iterator:
#             predictions = model(batch[0][1]).squeeze(1)
            
#             loss = criterion(predictions, batch.label)
#             acc = binary_accuracy(predictions, batch.label)

#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
        
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 5

best_valid_loss = float('inf')
losses = []
accuracies = []
for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
#     valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
        # torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    losses.append(train_loss)
    accuracies.append(train_acc)
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# plt.ylim(-0.1, 1.1)
# torch.save(model.state_dict(), 'tut1-model.pt')
# torch.save(model, 'tut1-sgd-ent.pt')

print(losses)
print(accuracies)
plt.plot(losses)
plt.plot(accuracies)


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

predict_sentiment(model, "This film is terrible")

predict_sentiment(model, "This film is very good")

