import math
import os

import evaluate
import nltk
import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, PLBartForConditionalGeneration

# Define the parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
max_input_length = 512
max_target_length = 50
batch_size = 8
seq_chunk_max_len = 300
context_len = (max_input_length - seq_chunk_max_len) // 2
context_statement_num = 5
mask_token_id = 50004
rouge = load_metric("rouge")
sacrebleu = evaluate.load("sacrebleu")
wandb_run_name = 'stage-2'
output_dir = '../output/models/' + wandb_run_name
input_train_file = "../datasets/stage2/train.csv"
input_valid_file = "../datasets/stage2/valid.csv"
tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")
model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-base")

args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=30,
    predict_with_generate=True,
    fp16=True,
    run_name=wandb_run_name,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)


def find_left_context_start(token_ids, start, end):
    if end >= len(token_ids):
        needed_statemens = 2 * context_statement_num
        search_end = start - 2 * context_len - 1
    else:
        needed_statemens = context_statement_num
        search_end = start - context_len - 1
    found = 0
    found_start = start
    for i in range(start - 2, max(-1, search_end), -1):
        if token_ids[i] == 65 or token_ids[i] == 43:
            found_start = i + 1
            found += 1
        if found == needed_statemens:
            return found_start if found_start >= 0 else 0
    return found_start if found_start >= 0 else 0


def find_right_context_end(token_ids, start, end):
    if start == 0:
        needed_statemens = 2 * context_statement_num
        search_end = end + 2 * context_len
    else:
        needed_statemens = context_statement_num
        search_end = end + context_len
    found = 0
    found_end = end
    for i in range(end, min(len(token_ids), search_end)):
        if token_ids[i] == 65 or token_ids[i] == 43:
            found_end = i + 1
            found += 1
        if found == needed_statemens:
            return found_end if found_end <= len(token_ids) else len(token_ids)
    return found_end if found_end <= len(token_ids) else len(token_ids)


def get_actual_input(token_ids):
    if mask_token_id not in token_ids:
        return 0, max_input_length
    start_idx = 0
    end_idx = len(token_ids)
    chunks_num = math.ceil(len(token_ids) / seq_chunk_max_len)
    chunks_len = math.ceil(len(token_ids) / chunks_num)
    for i in range(0, len(token_ids), chunks_len):
        if mask_token_id in token_ids[i: i + chunks_len]:
            start_idx = i
            end_idx = min(len(token_ids), i + chunks_len)
            break
    actual_start = find_left_context_start(token_ids, start_idx, end_idx)
    actual_end = find_right_context_end(token_ids, start_idx, end_idx)
    return actual_start, actual_end


def preprocess_function(batch):
    chunked_input_ids = []
    chunked_attention_masks = []
    inputs = tokenizer(batch["src"])
    for idx in range(len(inputs.input_ids)):
        input_id = inputs.input_ids[idx]
        attention_mask = inputs.attention_mask[idx]
        if len(input_id) <= 512:
            chunked_input_ids.append(input_id + [1] * (max_input_length - len(input_id)))
            chunked_attention_masks.append(attention_mask + [0] * (max_input_length - len(input_id)))
        else:
            input_id = input_id[:len(input_id) - 1]
            attention_mask = attention_mask[:len(attention_mask) - 1]
            chunk_start, chunk_end = get_actual_input(input_id)
            this_chunk_length = chunk_end - chunk_start
            if this_chunk_length == max_input_length:
                chunked_input_ids.append(input_id[chunk_start: chunk_end - 1] + [2])
                chunked_attention_masks.append(attention_mask[chunk_start: chunk_end - 1] + [1])
            else:
                chunked_input_ids.append(
                    input_id[chunk_start: chunk_end] + [2] + [1] * (max_input_length - this_chunk_length - 1))
                chunked_attention_masks.append(
                    attention_mask[chunk_start: chunk_end] + [1] + [0] * (max_input_length - this_chunk_length - 1))

    batch["input_ids"] = chunked_input_ids
    batch["attention_mask"] = chunked_attention_masks

    outputs = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=max_target_length)
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
                       batch["labels"]]

    return batch


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    decoded_preds_rouge = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels_rouge = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds_rouge, references=decoded_labels_rouge, use_stemmer=True)

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    result["bleu"] = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)["score"]

    string_to_compare = pd.DataFrame()
    string_to_compare["decoded_preds"] = ["".join(pred.split(" ")) for pred in decoded_preds]
    string_to_compare["decoded_labels"] = ["".join(label.split(" ")) for label in decoded_labels]
    string_to_compare["match"] = string_to_compare["decoded_preds"] == string_to_compare["decoded_labels"]
    result["accuracy"] = 100 * len(string_to_compare[string_to_compare["match"] == True]) / len(string_to_compare)

    return {k: round(v, 4) for k, v in result.items()}


if __name__ == '__main__':
    train_dataset = pd.read_csv(input_train_file, low_memory=False)
    train_dataset = train_dataset[['Position', 'LogStatement']]
    train_dataset.columns = ['src', 'target']

    train_dataset["src"] = train_dataset["src"].str.replace('<LOG>', '<mask>')
    train_dataset.dropna(axis=0, how='any', inplace=True)
    train_dataset = Dataset.from_pandas(train_dataset)

    val_dataset = pd.read_csv(input_valid_file)
    val_dataset = val_dataset[['Position', 'LogStatement']]
    val_dataset.columns = ['src', 'target']

    val_dataset["src"] = val_dataset["src"].str.replace('<LOG>', '<mask>')
    val_dataset.dropna(axis=0, how='any', inplace=True)
    val_dataset = Dataset.from_pandas(val_dataset)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)