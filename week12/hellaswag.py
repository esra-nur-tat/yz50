"""
HellaSwag eval
https://github.com/rowanz/hellaswag
Each example has a context and 4 possible endings, only one is correct (label).
The model doesn't generate anything, we just ask: for which ending is the
loss the lowest? Random guessing gets 25%.

example:
ctx: "A man is sitting on a roof. he"
endings: ["is using wrap to wrap a pair of skis.",
          "is ripping level tiles off.",          <- label = 1
          "is holding a rubik's cube.",
          "starts pulling up roofing on a roof."]

Used by train_gpt2.py at the end of training.
"""
import os
import urllib.request
import torch
from torch.nn import functional as F
import pyarrow.parquet as pq
import tiktoken

DATA_CACHE_DIR = "data_full"
url = "https://huggingface.co/datasets/allenai/hellaswag/resolve/main/data/validation-00000-of-00001.parquet"
enc = tiktoken.get_encoding("gpt2")

def load_examples(num_examples):
    # download the validation split once
    path = os.path.join(DATA_CACHE_DIR, "hellaswag_validation.parquet")
    if not os.path.exists(path):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    examples = pq.read_table(path).to_pylist()
    return examples[:num_examples]  # whole val set is 10042 examples, we use the first N

def render_example(example):
    # 4 rows: context + ending, padded to the same length
    # mask is 1 only on the ending tokens, that's where we measure the loss
    ctx_tokens = enc.encode(example["ctx"])
    tok_rows = []
    mask_rows = []
    for end in example["endings"]:
        end_tokens = enc.encode(" " + end)  # " " because GPT-2 tokens start with a space
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    label = int(example["label"])
    return tokens, mask, label

@torch.no_grad()
def evaluate(model, device, examples, name):
    model.eval()
    num_correct = 0
    for i, example in enumerate(examples, 1):
        tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)

        logits, _ = model(tokens)
        # loss at every position: logits at t predict the token at t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_tokens = tokens[:, 1:].contiguous()
        losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_tokens.view(-1), reduction='none')
        losses = losses.view(4, -1)

        # average loss only over the ending tokens (mask), not the context or padding
        # average instead of sum, otherwise long endings always lose
        shift_mask = mask[:, 1:]
        avg_loss = (losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
        pred = avg_loss.argmin().item()  # lowest loss = the ending the model finds most likely

        num_correct += int(pred == label)
        if i % 100 == 0:
            print(f"{name}: {num_correct}/{i} correct, acc: {num_correct/i:.4f}")
    return num_correct
