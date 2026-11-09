"""
FineWeb-Edu dataset (for pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads a small part of it, tokenizes with the GPT-2 tokenizer and saves
the tokens as shards in data_full/
Run once before train_gpt2.py:
$ python3 fineweb.py
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset

local_dir = "data_full"
shard_size = 1_000_000      # tokens per shard (his: 1e8)
total_tokens = 30_000_000   # his is 10B, 30M is enough for a 1 hour run on this Mac

os.makedirs(local_dir, exist_ok=True)

# streaming=True so we don't download the whole 10BT sample
fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token, separates the documents

def tokenize(doc):
    tokens = [eot]  # every document starts with <|endoftext|>
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)  # vocab is 50257 so uint16 is enough, half the size of int32

shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # buffer for the current shard
token_count = 0   # tokens in the current shard
total = 0         # tokens written in total

for doc in fw:
    tokens = tokenize(doc)
    pos = 0
    while pos < len(tokens) and total < total_tokens:
        # fill the current shard as much as possible, the rest of the doc goes to the next shard
        n = min(len(tokens) - pos, shard_size - token_count, total_tokens - total)
        all_tokens_np[token_count:token_count+n] = tokens[pos:pos+n]
        token_count += n
        pos += n
        total += n

        if token_count == shard_size or total == total_tokens:
            split = "val" if shard_index == 0 else "train"  # first shard is the validation set
            filename = os.path.join(local_dir, f"{split}_{shard_index:04d}.npy")
            np.save(filename, all_tokens_np[:token_count])
            print(f"saved {filename}: {token_count:,} tokens (total {total:,})")
            shard_index += 1
            token_count = 0

    if total >= total_tokens:
        break

# note: a document at the border of shard 0 and shard 1 can be split between val and train
