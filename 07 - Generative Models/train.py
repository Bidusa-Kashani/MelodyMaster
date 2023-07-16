import warnings

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, ClassLabel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from conditioned_attn_gru_decoder import ConditionedAttnGRUDecoder

warnings.filterwarnings("ignore")
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("train.csv")


def tokenize(batch):
    tokens = alephbert_tokenizer(batch['lyrics'], padding=True, truncation=True, max_length=512)
    tokens['labels'] = labels.str2int(batch['lyricist'])
    return tokens


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return {
        'top_1_accuracy': accuracy_score(labels, preds),
        'top_5_accuracy': top_k_accuracy_score(labels, pred.predictions, k=5),
        'precision': precision_score(labels, preds, average='macro'),
        'recall': recall_score(labels, preds, average='macro'),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
    }


lyricists = df['lyricist'].to_list()
lyrics = df['lyrics'].to_list()

X_train, X_val, y_train, y_val = train_test_split(lyrics, lyricists, test_size=0.15, stratify=lyricists)
train_df = pd.DataFrame(np.c_[X_train, y_train], columns=['lyrics', 'lyricist'])
val_df = pd.DataFrame(np.c_[X_val, y_val], columns=['lyrics', 'lyricist'])

labels = ClassLabel(names=df["lyricist"].unique().tolist())
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

alephbert_tokenizer = AutoTokenizer.from_pretrained('onlplab/alephbert-base')
alephbert = AutoModelForSequenceClassification.from_pretrained("onlplab/alephbert-base",
                                                               num_labels=len(df["lyricist"].unique()),
                                                               id2label={i: label for i, label in
                                                                         enumerate(labels.names)},
                                                               label2id={label: i for i, label in
                                                                         enumerate(labels.names)})

alephbert_embeddings = alephbert.bert.embeddings
for param in alephbert_embeddings.parameters():
    param.requires_grad = False

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

lyrics_vocab = set([word for lyric in train_df['lyrics'] for word in lyric.split()])
lyricist_tfidf = torch.zeros((len(labels.names), len(lyrics_vocab)))
for i, lyricist in enumerate(labels.names):
    lyricist_lyrics = train_df[train_df['lyricist'] == lyricist]['lyrics']
    lyricist_tfidf[i] = torch.Tensor(
        TfidfVectorizer(vocabulary=lyrics_vocab).fit_transform(lyricist_lyrics).toarray()[0])

lyricist_tfidf = lyricist_tfidf.to(device)

model = ConditionedAttnGRUDecoder(hidden_size=768, output_size=len(labels.names),
                                  alephbert_tokenizer=alephbert_tokenizer,
                                  alephbert_embeddings=alephbert_embeddings, lyricist_tfidf=lyricist_tfidf,
                                  device=device)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

model.train_all(train_loader, val_loader, optimizer, compute_metrics, epochs=10)

