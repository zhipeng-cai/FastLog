import os

import numpy as np
import pandas as pd
from datasets import load_metric, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification

from plbart_for_tokenclassification import PLBartForTokenClassification

# Define the parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_checkpoint = "uclanlp/plbart-base"
class_num = 2
batch_size = 8
max_seq_length = 512
label_all_tokens = True
wandb_run_name = "stage-1"
output_dir = "../output/models/" + wandb_run_name
metric = load_metric("metrics/sequval.py")
label_list = ["NO", "YES"]
input_train_file = "../datasets/stage1/train.csv"
input_valid_file = "../datasets/stage1/valid.csv"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
model = PLBartForTokenClassification.from_pretrained(model_checkpoint, num_labels=class_num)

args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    save_total_limit=2,
    weight_decay=0.01,
    run_name=wandb_run_name,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)


def get_word_ids(word_lists, input_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    word_idx = -1
    word_ids = []
    for token in tokens:
        if token in tokenizer.all_special_tokens:
            word_ids.append(None)
        elif token.startswith('▁') and word_lists[word_idx + 1].startswith(token[1:]):
            word_idx += 1
            word_ids.append(word_idx)
        else:
            word_ids.append(word_idx)

    return word_ids


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["src"], truncation=True, max_length=max_seq_length, padding='max_length',
                                 is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["target"]):
        word_ids = get_word_ids(examples['src'][i], tokenized_inputs['input_ids'][i])
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


if __name__ == '__main__':
    train_dataset = pd.read_csv(input_train_file, low_memory=False)
    train_dataset = train_dataset[['Input', 'Label']]
    train_dataset.columns = ['src', 'target']
    train_dataset.dropna(axis=0, how='any', inplace=True)
    train_dataset["src"] = train_dataset["src"].str.replace('java', 'Java')
    train_dataset["src"] = train_dataset["src"].str.replace('python', 'Python')
    train_dataset['src'] = train_dataset['src'].str.replace('…', '...')
    train_dataset["src"] = train_dataset["src"].apply(lambda x: x.split())
    train_dataset["target"] = train_dataset["target"].apply(lambda x: [int(num) for num in x.split()])
    train_dataset = Dataset.from_pandas(train_dataset)
    val_dataset = pd.read_csv(input_valid_file)
    val_dataset = val_dataset[['Input', 'Label']]
    val_dataset.columns = ['src', 'target']
    val_dataset.dropna(axis=0, how='any', inplace=True)
    val_dataset["src"] = val_dataset["src"].str.replace('java', 'Java')
    val_dataset["src"] = val_dataset["src"].str.replace('python', 'Python')
    val_dataset['src'] = val_dataset['src'].str.replace('…', '...')
    val_dataset["src"] = val_dataset["src"].apply(lambda x: x.split())
    val_dataset["target"] = val_dataset["target"].apply(lambda x: [int(num) for num in x.split()])
    val_dataset = Dataset.from_pandas(val_dataset)

    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)