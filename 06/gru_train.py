import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential, Softmax, GRU
from sklearn.model_selection import train_test_split

df = pd.read_csv("segmented_lyrics.csv")

df["lyrics_as_list"] = df["lyrics"].astype(str).str.strip().str.split(" ")
lyricists = df['lyricist']
# artists = df['artist'] # maybe we can try this one later.
lyrics = df['lyrics_as_list']

X_train, _, y_train, _ = train_test_split(lyrics, lyricists, test_size=0.15, stratify=lyricists)
X_train, X_test, y_train, y_test = train_test_split(lyrics, lyricists, test_size=0.15, stratify=lyricists)
train_df = pd.DataFrame(np.c_[X_train, y_train], columns=['lyrics', 'lyricist'])
val_df = pd.DataFrame(np.c_[X_test, y_test], columns=['lyrics', 'lyricist'])

with open('../words_list_w2v.txt', encoding='utf-8') as f:
    words = f.read().split('\n')
    # Removing the last word - an empty word
    words.pop()
vectors = np.load('../words_vectors_w2v.npy')

words = [w[3:] if len(w) > 3 and w[:3] in ['NN_', 'VB_', 'JJ_'] else w for w in words]

from collections import defaultdict, OrderedDict

def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)


dups_indx = [dup for dup in sorted(list_duplicates(words))]

words = list(OrderedDict.fromkeys(words))
vectors_without_dups = vectors.copy()
to_delete = []
for d in dups_indx:
    indices = sorted(d[1])
    vectors_without_dups[indices[0]] = np.mean(vectors[indices])
    to_delete = to_delete + indices[1:]
vectors = np.delete(vectors_without_dups, to_delete, 0)

w2v = dict(zip(words, vectors))
v2w_rep = dict(zip([tuple(v) for v in vectors], words))


def v2w(vector):
    return v2w_rep[tuple(vector)]


words_set = set(words)

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, lr=0.001, weight_decay=0.0,
                verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for i in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            batch = X_train[i:i + batch_size]
            labels = y_train[i:i + batch_size]
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (outputs.argmax(dim=1) == labels).sum().item()
        train_loss /= len(X_train)
        train_acc /= len(X_train)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch = X_val[i:i + batch_size]
                labels = y_val[i:i + batch_size]
                outputs = model(batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.argmax(dim=1) == labels).sum().item()
        val_loss /= len(X_val)
        val_acc /= len(X_val)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if verbose:
            print(f'Epoch {epoch + 1}/{epochs}, train loss: {train_loss:.3f}, train acc: {train_acc:.3f}, '
                  f'val loss: {val_loss:.3f}, val acc: {val_acc:.3f}')