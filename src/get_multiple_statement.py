import math
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, PLBartForConditionalGeneration

# Define the parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 1
beam_num = 10
input_max_length = 512
output_max_length = 50
seq_chunk_max_len = 300
context_len = (input_max_length - seq_chunk_max_len) // 2
context_statement_num = 5
mask_token_id = 50004
device = 'cuda'
used_position = 'Pred_Position10'
model_dir = '../output/models/stage2'
input_file = 'output/predictions/multiple-positions.csv'
output_dir = 'output/predictions'
output_file = output_dir + '/multiple-position10-statements.csv'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = PLBartForConditionalGeneration.from_pretrained(model_dir)
model.to(device)
model.eval()
torch.manual_seed(0)


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


def post_process_punctuation(logsta):
    if logsta == None:
        return ""
    text_len = len(logsta)
    insert_space_idxs = []
    is_inside_quotes = False
    for i in range(text_len):
        if logsta[i] == '"':
            is_inside_quotes = not is_inside_quotes
        if is_inside_quotes:
            continue
        if logsta[i] == '.' or logsta[i] == ',':
            insert_space_idxs.append(i)
    for i in reversed(insert_space_idxs):
        logsta = logsta[:i] + " " + logsta[i:]

    return logsta


def get_log_level(logsta):
    for token in logsta.split():
        if token in ["trace", "debug", "info", "warn", "error", "fatal"]:
            return token
    return ""


def get_log_message(logsta):
    left_idx = logsta.find('(')
    right_idx = logsta.rfind(')')

    if left_idx == -1:
        start_idx = 0
    else:
        start_idx = left_idx
    if right_idx == -1:
        end_idx = len(logsta) - 1
    else:
        end_idx = right_idx + 1

    message = logsta[start_idx:end_idx].strip()
    return message


if __name__ == '__main__':
    input_texts = pd.read_csv(input_file)
    input_texts = input_texts[used_position]
    input_texts.fillna("", inplace=True)
    input_texts = input_texts.str.replace('<LOG>', '<mask>')
    total_len = len(input_texts)

    predictions = []
    with torch.no_grad():
        cache = []

        for index, input_text in input_texts.items():
            if index > 0 and index % 100 == 0:
                print("Progress: {}/{}".format(index, total_len), len(predictions), predictions[-1])

            cache.append(input_text)

            tokenized_text = tokenizer(cache, return_tensors='pt')
            source_ids = tokenized_text['input_ids'][0]
            source_mask = tokenized_text['attention_mask'][0]
            chunked_input_ids = []
            chunked_attention_masks = []

            if len(source_ids) <= 512:
                chunked_input_ids.append(torch.tensor(source_ids.tolist() + [1] * (input_max_length - len(source_ids))))
                chunked_attention_masks.append(
                    torch.tensor(source_mask.tolist() + [0] * (input_max_length - len(source_ids))))
            else:
                source_ids = source_ids[:len(source_ids) - 1]
                source_mask = source_mask[:len(source_mask) - 1]
                chunk_start, chunk_end = get_actual_input(source_ids)
                this_chunk_length = chunk_end - chunk_start
                if this_chunk_length == input_max_length:
                    chunked_input_ids.append(torch.tensor(source_ids[chunk_start: chunk_end - 1].tolist() + [2]))
                    chunked_attention_masks.append(torch.tensor(source_mask[chunk_start: chunk_end - 1].tolist() + [1]))
                else:
                    chunked_input_ids.append(torch.tensor(source_ids[chunk_start: chunk_end].tolist() + [2] + [1] * (
                            input_max_length - this_chunk_length - 1)))
                    chunked_attention_masks.append(torch.tensor(
                        source_mask[chunk_start: chunk_end].tolist() + [1] + [0] * (
                                input_max_length - this_chunk_length - 1)))

            chunked_input_ids = torch.stack(chunked_input_ids).to(device, dtype=torch.long)
            chunked_attention_masks = torch.stack(chunked_attention_masks).to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=chunked_input_ids,
                attention_mask=chunked_attention_masks,
                max_length=output_max_length,
                num_beams=beam_num,
                num_return_sequences=beam_num,
                early_stopping=True
            )

            generated_texts = []
            for i, generated_id in enumerate(generated_ids):
                generated_texts.append(tokenizer.decode(generated_id, skip_special_tokens=True))
                if (i + 1) % beam_num == 0:
                    generated_texts = [post_process_punctuation(text) for text in generated_texts]
                    predictions.append(generated_texts)
                    generated_texts = []

            cache = []

    predictions = pd.DataFrame(predictions)
    predictions.to_csv(output_file)
    print(predictions)