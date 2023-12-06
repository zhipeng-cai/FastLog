import math
import os

import pandas as pd
import torch
from transformers import AutoTokenizer

from plbart_for_tokenclassification import PLBartForTokenClassification

# Define the parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 1
model_max_length = 512
seq_chunk_max_len = 300
context_len = (model_max_length - seq_chunk_max_len) // 2
context_statement_num = 5
device = 'cuda'
model_dir = '../output/models/stage1'
input_file = '../datasets/stage1/original-test.csv'
src_file = "../datasets/stage2/original-test.csv"
output_dir = '../output/predictions'
output_file = output_dir + '/positions.csv'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
model = PLBartForTokenClassification.from_pretrained(model_dir)
model.to(device)
model.eval()
torch.manual_seed(0)


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


if __name__ == '__main__':
    input_texts = pd.read_csv(input_file)
    input_texts['Input'] = input_texts['Input'].str.replace('java', 'Java')
    input_texts['Input'] = input_texts['Input'].str.replace('python', 'Python')
    input_texts = input_texts['Input'].apply(lambda x: x.split())
    total_len = len(input_texts)

    predictions = []
    max_probabilitys = []
    token_counts = []

    with torch.no_grad():
        cache = []

        for index, input_text in input_texts.items():
            if index > 0 and index % 100 == 0:
                print("Progress: {}/{}".format(index, total_len), len(predictions), predictions[-1], token_counts[-1],
                      max_probabilitys[-1])

            cache.append(input_text)

            tokenized_text = tokenizer(cache, return_tensors='pt', is_split_into_words=True)
            source_ids = tokenized_text['input_ids'][0]
            source_mask = tokenized_text['attention_mask'][0]
            token_counts.append(len(source_ids))

            if len(source_ids) <= model_max_length:
                chunked_input_ids = [torch.tensor(source_ids.tolist() + [1] * (model_max_length - len(source_ids)))]
                chunked_attention_mask = [
                    torch.tensor(source_mask.tolist() + [0] * (model_max_length - len(source_ids)))]
                chunked_valid_start_idxs = [0]
                chunked_valid_end_idxs = [model_max_length]
                chunks_len = model_max_length
            else:
                source_ids = source_ids[:len(source_ids) - 1]
                source_mask = source_mask[:len(source_mask) - 1]
                # 1. fixed chunks length:
                # chunks_num = math.ceil(len(source_ids) / seq_chunk_max_len)
                # chunks_len = seq_chunk_max_len

                # 2. average chunks length:
                # chunks_num = math.ceil(len(source_ids) / seq_chunk_max_len)
                # chunks_len = math.ceil(len(source_ids) / chunks_num)

                # 3. average chunks length with left & right context
                chunks_num = math.ceil(len(source_ids) / seq_chunk_max_len)
                chunks_len = math.ceil(len(source_ids) / chunks_num)

                chunked_input_ids = []
                chunked_attention_mask = []
                chunked_valid_start_idxs = []
                chunked_valid_end_idxs = []
                # 2. average chunks length:
                # for i in range(0, len(source_ids), chunks_len):
                #     chunk_start = max(0, i - context_len)
                #     chunk_end = min(len(source_ids), i + chunks_len + context_len)
                #     chunked_input_ids.append(source_ids[chunk_start: chunk_end])
                #     chunked_attention_mask.append(source_mask[chunk_start: chunk_end])

                # 3. average chunks length with left & right context
                for i in range(0, len(source_ids), chunks_len):
                    chunk_start = find_left_context_start(source_ids, i, i + chunks_len)
                    chunk_end = find_right_context_end(source_ids, i, i + chunks_len)
                    this_chunk_length = chunk_end - chunk_start
                    if this_chunk_length == model_max_length:
                        input_ids = torch.tensor(source_ids[chunk_start: chunk_end - 1].tolist() + [2])
                        attention_masks = torch.tensor(source_mask[chunk_start: chunk_end - 1].tolist() + [1])
                    else:
                        input_ids = torch.tensor(source_ids[chunk_start: chunk_end].tolist() + [2] + [1] * (
                                model_max_length - this_chunk_length - 1))
                        attention_masks = torch.tensor(source_mask[chunk_start: chunk_end].tolist() + [1] + [0] * (
                                model_max_length - this_chunk_length - 1))

                    chunked_input_ids.append(input_ids)
                    chunked_attention_mask.append(attention_masks)
                    chunked_valid_start_idxs.append(i - chunk_start)
                    chunked_valid_end_idxs.append(chunk_end - i - chunks_len)

            chunked_input_ids = torch.stack(chunked_input_ids).to(device, dtype=torch.long)
            chunked_attention_mask = torch.stack(chunked_attention_mask).to(device, dtype=torch.long)

            outputs = model(chunked_input_ids, attention_mask=chunked_attention_mask).logits

            probabilities = torch.nn.functional.softmax(outputs, dim=2)

            split_size, seq_len, num_labels = probabilities.shape
            last_class_probabilities = probabilities[:, :, -1].tolist()

            word_list = cache[0]
            tokenized_word_ids = get_word_ids(word_list, source_ids)
            valid_word_idxs = [j for j in range(len(word_list)) if word_list[j] in ["{", ";", "}", ":"]]
            valid_token_idxs = [j for j in range(len(tokenized_word_ids)) if tokenized_word_ids[j] in valid_word_idxs]

            valid_probabilities = []
            # 2. average chunks length:
            # for j in valid_token_idxs:
            #     valid_probabilities.append([last_class_probabilities[j // chunks_len][j % chunks_len], tokenized_word_ids[j]])

            # 3. average chunks length with left & right context
            for j in valid_token_idxs:
                split_idx = j // chunks_len
                seq_idx = chunked_valid_start_idxs[split_idx] + j % chunks_len
                valid_probabilities.append([last_class_probabilities[split_idx][seq_idx], tokenized_word_ids[j]])

            sorted_valid_probabilities = sorted(valid_probabilities, key=lambda s: s[0], reverse=True)

            predictions.append(sorted_valid_probabilities[0][1])
            max_probabilitys.append(sorted_valid_probabilities[0][0])

            cache = []

    data = pd.read_csv(src_file)
    inputs = data["Input"].apply(lambda x: x.split()).tolist()
    pred_mask_position = []

    for i in range(len(predictions)):
        raw_input = " ".join(inputs[i])
        inputs[i].insert(predictions[i] + 1, "<LOG>")
        pred_mask_position.append(" ".join(inputs[i]))

    new_data = pd.DataFrame()
    new_data["Position"] = predictions
    new_data["Probabilitys"] = max_probabilitys
    new_data["token_counts"] = token_counts
    new_data["Input"] = data["Input"]
    new_data["True_Position"] = data["Position"]
    new_data["Pred_Position"] = pred_mask_position
    new_data.to_csv(output_file)
    print(new_data)