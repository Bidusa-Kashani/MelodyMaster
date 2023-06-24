import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, \
    get_linear_schedule_with_warmup, default_data_collator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("train.csv")

lyricists = df['lyricist'].to_list()
lyrics = df['lyrics'].to_list()

X_train, X_val, y_train, y_val = train_test_split(lyrics, lyricists, test_size=0.15, stratify=lyricists)
train_df = pd.DataFrame(np.c_[X_train, y_train], columns=['lyrics', 'lyricist'])
val_df = pd.DataFrame(np.c_[X_val, y_val], columns=['lyrics', 'lyricist'])

#train_df['lyrics'] = train_df['lyrics'].apply(lambda x: ' '.join(x.split()[:512]))
#val_df['lyrics'] = val_df['lyrics'].apply(lambda x: ' '.join(x.split()[:512]))

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

alephbert_tokenizer = AutoTokenizer.from_pretrained('onlplab/alephbert-base')
alephbert = AutoModelForSequenceClassification.from_pretrained("onlplab/alephbert-base",
                                                               num_labels=len(df["lyricist"].unique()))
alephbert.config.id2label = {i: label for i, label in enumerate(df["lyricist"].unique())}
alephbert.config.label2id = {label: i for i, label in enumerate(df["lyricist"].unique())}

# TODO: set freezed / non-freezed layers
# alephbert.classifier.out_features = len(df['lyricist'].unique())
for p in alephbert.bert.embeddings.parameters():
    p.requires_grad = False

print(alephbert)
args = TrainingArguments(
    "output",
    evaluation_strategy="steps",
    eval_steps=10,
    save_steps=100,
    save_total_limit=5,
    load_best_model_at_end=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    learning_rate=1e-3,
    weight_decay=0.01,
    warmup_steps=int(1e3),
    logging_dir="logs",
    logging_steps=100,
    fp16=True,
    seed=42,
)

optimizer = AdamW(alephbert.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                               num_training_steps=args.max_steps)


trainer = Trainer(
    model=alephbert,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=alephbert_tokenizer,
    data_collator=default_data_collator,
    optimizers=(optimizer, lr_scheduler),
)

trainer.train()
