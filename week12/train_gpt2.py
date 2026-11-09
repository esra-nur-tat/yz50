from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math 
import time
import inspect

# Multi-head causal self-attention (diff vs week 8: q, k, v are computed in one single linear layer)
class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)  # one layer produces q, k and v together
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)    # output projection
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv= self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)  # split the combined output back into q, k, v
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention, replaces the manual att + mask from week 11
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back together
        y = self.c_proj(y)
        return y

# Feed-forward network inside each block
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')  # diff vs week 8: GPT-2 uses GELU instead of ReLU
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        
# One transformer block: attention + MLP, each with a residual connection
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp= MLP(config)

    def forward(self,x):
        # diff vs week 8: LayerNorm comes before attn/mlp (pre-norm), not after
        x= x+ self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT-2 (124M) hyperparameters
@dataclass
class GPTConfig:
    block_size: int= 1024    # max sequence length
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    n_layer: int = 12        # number of transformer blocks
    n_head: int=12           # number of attention heads
    n_embd: int = 768        # embedding size

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # the stack of blocks
            ln_f = nn.LayerNorm(config.n_embd),  # final layer norm
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # maps features to vocab logits
        #weight sharing schema
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std=0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):  
            std *= (2 * self.config.n_layer) ** -0.5 #2 because 1 for mlp 1 for attn
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)   
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # add token and position embeddings together
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:  # pass through every transformer block
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # cross-entropy loss between predicted logits and the next-token targets
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available.
        # fused=True is a CUDA-only kernel, so on this Mac device_type is 'mps'
        # and use_fused stays False.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 
        config_args['block_size'] = 1024 
        
        config = GPTConfig(**config_args)
        model = GPT(config)  # our own model with the same shapes
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        # HF stores these as Conv1D (transposed) weights, so we must transpose them to fit our nn.Linear layers
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
            
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

#-------- Data Loader --------
# week 11 read input.txt, now the tokens come from the FineWeb-Edu shards
# made by fineweb.py (data_full/train_0001.npy ... train_0029.npy, val_0000.npy)
import os
import numpy as np
import tiktoken
import hellaswag

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int64)  # uint16 on disk, cross_entropy wants long
    return torch.tensor(npt, dtype=torch.long)

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        data_root = "data_full"
        shards = os.listdir(data_root)
        shards = [s for s in shards if s.startswith(split) and s.endswith('.npy')]
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no shards found for split {split}, run fineweb.py first"
        print(f"found {len(self.shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # start from the first shard (val always uses the same batches this way)
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # if the next batch doesn't fit, move to the next shard
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) #inputs
        y = (buf[1:]).view(B, T) #targets
        self.current_position += B*T
        return x, y

# --- Batch config ---
# His total_batch_size is 524288 (B=64, T=1024). With B=32 T=128 (week 11) one step
# took ~4.7s, so for a 1 hour run I went smaller to get more optimizer steps.
B = 4
T = 128
grad_accum_steps = 8
total_batch_size = B * T * grad_accum_steps   # 4096 tokens per optimizer step
print(f"total batch size: {total_batch_size}, grad accum steps: {grad_accum_steps}")

device = 'mps'
torch.manual_seed(1337)

train_loader = DataLoaderLite(B=B, T=T, split='train')
val_loader = DataLoaderLite(B=B, T=T, split='val')

model = GPT(GPTConfig())  # randomly initialized model
model.to(device)

# training is limited by time (1 hour) and not by steps, so the lr schedule uses seconds
max_lr = 6e-4
min_lr = max_lr * 0.1
train_seconds = 60 * 60
warmup_seconds = 60
eval_every_seconds = 5 * 60   # val loss + a sample every 5 minutes
val_loss_steps = 20

def get_lr(elapsed):
    # 1) linear warmup
    if elapsed < warmup_seconds:
        return max_lr * max(elapsed / warmup_seconds, 1 / warmup_seconds)
    # 2) cosine decay down to min_lr until the end of the hour
    decay_ratio = min((elapsed - warmup_seconds) / (train_seconds - warmup_seconds), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device)
enc = tiktoken.get_encoding('gpt2')

def evaluate_and_sample(step, tokens_seen):
    model.eval()
    # --- validation loss ---
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            val_loss_accum += loss.item() / val_loss_steps

    # --- text generation with top-k sampling (same as week 10) ---
    # same seed every time so we can compare how the text changes during training
    torch.manual_seed(42)
    x = torch.tensor([enc.encode("Hello, I'm a language model,")], device=device)
    with torch.no_grad():
        for _ in range(25):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(x)
            logits = logits[:, -1, :].float()  # only the last position predicts the next token
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    print(f"validation | step {step} | tokens {tokens_seen:,} | val loss {val_loss_accum:.4f}")
    print(enc.decode(x[0].tolist()))
    model.train()
    return val_loss_accum

# for the plot
train_tokens, train_losses = [], []
val_tokens, val_losses = [], []

step = 0
tokens_seen = 0
training_time = 0.0   # only the optimizer steps, eval and sampling are not counted

val_loss = evaluate_and_sample(step, tokens_seen)  # random model, should be around -ln(1/50257) = 10.8
val_tokens.append(0)
val_losses.append(val_loss)
next_eval = eval_every_seconds

start = time.perf_counter()
while time.perf_counter() - start < train_seconds:
    lr = get_lr(time.perf_counter() - start)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    torch.mps.synchronize()
    t0 = time.perf_counter()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # backward() sums the gradients, dividing makes it the mean again
        loss_accum += loss.detach().float().item()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.mps.synchronize()  # wait for the GPU, otherwise dt is wrong
    dt = time.perf_counter() - t0

    training_time += dt
    step += 1
    tokens_seen += total_batch_size
    train_tokens.append(tokens_seen)
    train_losses.append(loss_accum)
    print(f"step {step:4d} | loss {loss_accum:.4f} | lr {lr:.6f} | norm {norm:.4f} | dt {dt:.2f}s | tok/sec {total_batch_size/dt:.0f} | {tokens_seen:,} tokens")

    if time.perf_counter() - start >= next_eval:
        val_loss = evaluate_and_sample(step, tokens_seen)
        val_tokens.append(tokens_seen)
        val_losses.append(val_loss)
        next_eval += eval_every_seconds

if val_tokens[-1] != tokens_seen:
    # one last validation at the end
    val_loss = evaluate_and_sample(step, tokens_seen)
    val_tokens.append(tokens_seen)
    val_losses.append(val_loss)

print("-" * 70)
print(f"steps            : {step}")
print(f"tokens seen      : {tokens_seen:,}")
print(f"training time    : {training_time:.1f} s")
print(f"throughput       : {tokens_seen/training_time:.1f} tok/sec")
print(f"final train loss : {loss_accum:.4f}")
print(f"final val loss   : {val_loss:.4f}")
print("-" * 70)

os.makedirs("log", exist_ok=True)
torch.save(model.state_dict(), "log/model.pt")  # ~500MB, not pushed to github

# --- train / val loss plot ---
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))
plt.plot(train_tokens, train_losses, alpha=0.5, label='Train loss')
plt.plot(val_tokens, val_losses, marker='o', label='Validation loss')
plt.xlabel('tokens seen')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss.png', dpi=150)

# --- HellaSwag: our model vs OpenAI GPT-2 on the same examples ---
num_examples = 1000  # whole val set takes too long, first 1000 is enough
examples = hellaswag.load_examples(num_examples)

ours = hellaswag.evaluate(model, device, examples, "our model")
del model
torch.mps.empty_cache()

model_gpt2 = GPT.from_pretrained('gpt2')
model_gpt2.to(device)
theirs = hellaswag.evaluate(model_gpt2, device, examples, "OpenAI GPT-2")

print(f"HellaSwag (first {num_examples} val examples)")
print(f"our model    : {ours}/{num_examples} = {ours/num_examples:.3f}")
print(f"OpenAI GPT-2 : {theirs}/{num_examples} = {theirs/num_examples:.3f}")
print(f"random guess : 0.25")


"""1 hour run, Apple M5 (mps), bf16, B=4 T=128 grad_accum=8
val dataset: 1M tokens, train dataset: 29M tokens

step    1 | loss 11.0158 | lr 0.000010 | 1.14s | 3585 tok/s | 4,096 token
step    2 | loss 10.5554 | lr 0.000011 | 1.09s | 3754 tok/s | 8,192 token
step    3 | loss 10.3139 | lr 0.000022 | 0.89s | 4626 tok/s | 12,288 token
...
step 4123 | loss 5.6574 | lr 0.000060 | 0.85s | 4802 tok/s | 16,887,808 token
step 4124 | loss 5.7089 | lr 0.000060 | 0.85s | 4800 tok/s | 16,891,904 token
step 4125 | loss 6.1435 | lr 0.000060 | 0.85s | 4802 tok/s | 16,896,000 token
----------------------------------------------------------------------
steps            : 4125
tokens seen      : 16,896,000
training time    : 3591.0 s
throughput       : 4705.1 tok/sec
final train loss : 6.1435
final val loss   : 5.9097
----------------------------------------------------------------------
"""

"""samples during training (same seed every time)
step 0, tokens 0, val loss 11.0040
Hello, I'm a language model, logpleasefamilylv teams comparisonswitch LD enjoyableUF Sergey mockyonyon Facilities ministryager Fnissors Emirisoftorganismsided Radioec

step 345, tokens 1,413,120, val loss 6.9384
Hello, I'm a language model, where the area,
The body. We do all to the most common and most types about their important.
In the

step 666, tokens 2,727,936, val loss 6.6958
Hello, I'm a language model, or the
These does not to the United States'
We have be been aware of such in the water. In the

step 990, tokens 4,055,040, val loss 6.6109
Hello, I'm a language model, a certain sense of her own way, the study could be very more likely to establish, according to be. This's it

step 1332, tokens 5,455,872, val loss 6.4357
Hello, I'm a language model, a world?
An important
The
I)
and you're the whole words – we have the good, you

step 1685, tokens 6,901,760, val loss 6.3457
Hello, I'm a language model, as a more than how to make the “To see the “We.” is a few years of the

step 2033, tokens 8,327,168, val loss 6.2572
Hello, I'm a language model, we might know, our own work, we don’t know, but like you be the important way. We will

step 2379, tokens 9,744,384, val loss 6.1471
Hello, I'm a language model, which is not much enough to see how you should in the classroom. If the way for other things, you want to become

step 2726, tokens 11,165,696, val loss 6.0797
Hello, I'm a language model, it is only because what it was the goal of people. What is many people is in, the same? How we don

step 3075, tokens 12,595,200, val loss 6.0161
Hello, I'm a language model, or after one thing from the students to understand of someone.
The learning is no reason “I will’ve

step 3424, tokens 14,024,704, val loss 5.9814
Hello, I'm a language model, in the course of some of my students’s parents’d be made out but it’s the problem

step 3774, tokens 15,458,304, val loss 5.9325
Hello, I'm a language model, the problem’s a new, so is this is to be. I mean a difference, I will be the best

step 4125, tokens 16,896,000, val loss 5.9097
Hello, I'm a language model, the right to your favorite. This will do not exist, then, the most successful work makes you not only to make a
"""
# at step 0 it is random tokens, then common words and punctuation come first,
# later short english phrases ("in the classroom", "the students"), it picks up the
# educational style of fineweb-edu. Still no real meaning in the sentences.

"""HellaSwag (first 1000 val examples)
our model: 25/100 49/200 71/300 99/400 125/500 153/600 176/700 207/800 236/900 268/1000
OpenAI GPT-2: 32/100 62/200 95/300 128/400 168/500 212/600 249/700 277/800 315/900 336/1000

our model    : 268/1000 = 0.268
OpenAI GPT-2 : 336/1000 = 0.336
random guess : 0.25
"""
# Why the gap: our model saw ~17M tokens, Karpathy's run saw 10B and the OpenAI GPT-2
# was trained on WebText (~100B tokens seen). Also T=128 instead of 1024 and a much
# smaller batch (4096 tokens vs 0.5M), so the gradients are noisy. 26.8% is only a bit
# above random 25%, with 1000 examples that difference is close to noise.
