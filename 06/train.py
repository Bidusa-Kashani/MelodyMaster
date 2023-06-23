import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import BertModel, BertTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
alephbert = BertModel.from_pretrained('onlplab/alephbert-base')


# TODO: set freezed / non-freezed layers

df = pd.read_csv("train.csv")

lyricists = df['lyricist']
lyrics = df['lyrics']

X_train, X_val, y_train, y_val = train_test_split(lyrics, lyricists, test_size=0.15, stratify=lyricists)
train_df = pd.DataFrame(np.c_[X_train, y_train], columns=['lyrics', 'lyricist'])
val_df = pd.DataFrame(np.c_[X_val, y_val], columns=['lyrics', 'lyricist'])

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

optimizer = AdamW(alephbert.parameters(), lr=1e-3, weight_decay=0.01)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1e3, num_training_steps=1e4)


args = {
    "output_dir": "output",
    "overwrite_output_dir": True,
    "do_train": True,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 8,
    "fp16": True,
    "logging_dir": "logs",
    "logging_steps": 100,
    "evaluation_strategy": "steps",
    "eval_steps": 10,
    "save_steps": 100,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "seed": 42,
    "full_determinism": False
}

alephbert.to(device)

trainer = Trainer(
    model=alephbert,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=alephbert_tokenizer,
    optimizers=[optimizer, lr_scheduler],
    args=TrainingArguments(**args)
)

trainer.train()