import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import Dataset, ClassLabel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AdamW, \
    get_linear_schedule_with_warmup, default_data_collator, EarlyStoppingCallback

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

alephbert.classifier = nn.Sequential(
    nn.Linear(in_features=768, out_features=384, bias=True),
    nn.Tanh(),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=384, out_features=labels.num_classes, bias=True)
)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

for p in alephbert.bert.embeddings.parameters():
    p.requires_grad = False

for p in alephbert.bert.encoder.parameters():
    p.requires_grad = False

for name, module in alephbert.bert.named_modules():
    if isinstance(module, torch.nn.Dropout) and not (
            "embeddings" in name or "encoder.layer." in name):
        module.p = 0.5

alephbert.to(device)

args = TrainingArguments(
    "output7",
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    eval_steps=5,
    save_steps=5,
    logging_steps=5,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="top_5_accuracy",
    num_train_epochs=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir="logs7",
    fp16=False,
    seed=42,
)

BATCHES_PER_EPOCH = len(train_dataset) // args.per_device_train_batch_size + (
    1 if len(train_dataset) % args.per_device_train_batch_size else 0)

optimizer = AdamW(alephbert.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=args.warmup_ratio * args.num_train_epochs * BATCHES_PER_EPOCH,
                                               num_training_steps=args.num_train_epochs * BATCHES_PER_EPOCH)

trainer = Trainer(
    model=alephbert,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=alephbert_tokenizer,
    data_collator=default_data_collator,
    optimizers=(optimizer, lr_scheduler),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3000)]
)

trainer.train(resume_from_checkpoint=False)
