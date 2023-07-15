import warnings

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, ClassLabel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score, \
    classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AdamW, \
    get_linear_schedule_with_warmup, default_data_collator

warnings.filterwarnings("ignore")
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "output5"

original_train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


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


def save_eval_data(dataset: Dataset, dataset_name: str):
    with torch.no_grad():
        preds = trainer.predict(dataset)
    metrics = pd.DataFrame(compute_metrics(preds), index=[0])
    metrics.to_csv(f"eval_results/{dataset_name}_metrics.csv", index=False)
    report = classification_report(preds.label_ids, preds.predictions.argmax(-1), target_names=labels.names,
                                   output_dict=True)
    report.pop("accuracy")
    report.pop("macro avg")
    report.pop("weighted avg")
    report = pd.DataFrame(report).transpose()
    report.to_csv(f"eval_results/{dataset_name}_classification_report.csv", index=True)
    cm = confusion_matrix(preds.label_ids, preds.predictions.argmax(-1))
    np.save(f"eval_results/{dataset_name}_confusion_matrix.npy", cm)


lyricists = original_train_df['lyricist'].to_list()
lyrics = original_train_df['lyrics'].to_list()

X_train, X_val, y_train, y_val = train_test_split(lyrics, lyricists, test_size=0.15, stratify=lyricists)
train_df = pd.DataFrame(np.c_[X_train, y_train], columns=['lyrics', 'lyricist'])
val_df = pd.DataFrame(np.c_[X_val, y_val], columns=['lyrics', 'lyricist'])
test_df = test_df[["lyrics", "lyricist"]]

labels = ClassLabel(names=original_train_df["lyricist"].unique().tolist())
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

alephbert_tokenizer = AutoTokenizer.from_pretrained('onlplab/alephbert-base')
alephbert = AutoModelForSequenceClassification.from_pretrained("onlplab/alephbert-base",
                                                               num_labels=len(original_train_df["lyricist"].unique()),
                                                               id2label={i: label for i, label in
                                                                         enumerate(labels.names)},
                                                               label2id={label: i for i, label in
                                                                         enumerate(labels.names)})

for name, module in alephbert.bert.named_modules():
    if isinstance(module, torch.nn.Dropout) and (
            not ("embeddings" in name or "encoder.layer." in name) or "encoder.layer.11" in name):
        module.p = 0.5

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

alephbert.to(device)

args = TrainingArguments(
    "output5",
    max_steps=0,
    # do_train=True,
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
    # logging_dir="logs5",
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
    compute_metrics=compute_metrics
)

trainer.train(resume_from_checkpoint=True)

alephbert.eval()
save_eval_data(train_dataset, "train")
save_eval_data(val_dataset, "val")
save_eval_data(test_dataset, "test")
