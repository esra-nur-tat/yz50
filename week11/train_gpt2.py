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

        # lower-triangular mask so a token can only attend to past tokens (causal)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv= self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)  # split the combined output back into q, k, v
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # scaled attention scores
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # hide future tokens
        att = F.softmax(att, dim=-1)
        y = att @ v  # weighted sum of values
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
        if hasattr(module, 'NANOGPT_SCLAE_INIT'):  
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
"""
# --- Build one small batch of tokens from a text file ---
import tiktoken
enc = tiktoken.get_encoding('gpt2')  # GPT-2 tokenizer
with open('input.txt', 'r') as f:
    text = f.read()
text= text[:1000]
tokens = enc.encode(text)
B, T= 4, 32  # batch size and sequence length
buf = torch.tensor(tokens[:B*T +1])
x= buf [:-1].view(B, T)   # inputs
y = buf[1:].view(B, T)    # targets are the inputs shifted by one token
"""
#-------- Data Loader --------
import tiktoken
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) #inputs
        y = (buf[1:]).view(B, T)#targets

        self.current_position += B*T

        if self.current_position + (B*T +1) > len(self.tokens):
            self.current_position = 0

        return x,y 

# --- Batch config  ---
# He uses total_batch_size = 524288 (2**19) with B=64, T=1024 on 8x A100 80GB.
# That needs ~80GB per GPU; this Mac has 24GB unified, so B/T are scaled down
# and total_batch_size is reduced to keep one optimizer step a few seconds.
total_batch_size = 32768   # tokens per optimizer step (his: 524288)
B = 32                     # micro batch size   (his: 64)
T = 128                    # sequence length    (his: 1024)
assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)   # NOT used yet, see note below
print(f"micro batch: B={B} T={T} -> {B*T} tokens per step")
print(f"calculated gradient accumulation steps {grad_accum_steps} "
      f"-> micro-steps to reach total_batch_size={total_batch_size})")

train_loader = DataLoaderLite(B=B, T=T)

model = GPT(GPTConfig())  # randomly initialized model
model.to('mps')
#model = torch.compile(model)
torch.manual_seed(1337)

max_lr = 3e-4
min_lr= max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it<warmup_steps:
        return max_lr * (it+1) / warmup_steps

    if it > max_steps:  
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type='mps')

step_times = []                                   # seconds per step
# No gradient accumulation right now: one batch per optimizer step, so the
# token count MUST be B*T. Using total_batch_size here would overstate
# throughput by exactly grad_accum_steps (it reported 50341 instead of ~6292).
tokens_per_step = B * T

for step in range(200):
    t0 = time.perf_counter()
    optimizer.zero_grad()  
    loss_accum = 0.0
    for micro_steps in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to('mps'), y.to('mps')
        with torch.autocast(device_type='mps', dtype=torch.bfloat16):
            logits, loss = model(x, y)  # forward pass + loss
        loss = loss/grad_accum_steps # We divide by grad_accum_steps because backward() sums the micro-batch gradients, but a real batch averages them — the division turns the accumulated sum back into the correct average
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

  
    # MPS dispatches work asynchronously: without this sync we would only be
    # timing how long it took to *queue* the kernels, not to run them.
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    step_times.append(dt)

    # .item() is free here, the sync above already waited for the GPU
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, "
          f"dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

# --- Timing summary ---
# The first few steps include one-off costs (kernel compilation, allocator
# warmup), so they are excluded from the steady-state numbers below.
WARMUP = 5
steady = step_times[WARMUP:] or step_times
steady_sorted = sorted(steady)
median = steady_sorted[len(steady_sorted) // 2]
mean = sum(steady) / len(steady)

print("-" * 70)
print(f"steps                : {len(step_times)}")
print(f"total wall time      : {sum(step_times):.2f} s")
print(f"first step           : {step_times[0]*1000:7.2f} ms  (includes warmup)")
print(f"steady mean          : {mean*1000:7.2f} ms  (steps {WARMUP}+)")
print(f"steady median        : {median*1000:7.2f} ms")
print(f"fastest / slowest    : {min(steady)*1000:7.2f} / {max(steady)*1000:7.2f} ms")
print(f"throughput           : {tokens_per_step/median:8.1f} tok/sec  (from median)")
print("-" * 70)

import sys; sys.exit(0)

# --- Text generation with top-k sampling ---
model.eval()
num_run_sequences=5
max_length = 30

torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)
while x.size(1)< max_length:  # keep generating until we reach max_length

    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :]  # only the last position predicts the next token
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # keep the 50 most likely tokens
        ix= torch.multinomial(topk_probs, 1)        # sample one of them
        xcol= torch.gather(topk_indices, -1, ix)    # map back to the real token id

        x=torch.cat((x,xcol), dim=1)  # append the new token and repeat

for i in range(num_run_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(decoded)
# I couldn't implement TF32 (no Tensor Cores on Apple GPU), flash attention (fused kernel is CUDA-only, MPS uses fallback)
"""
Hello, I have midterm exams ive been getting them because I want them to be right. I'm a big baby who grew up in an area
Hello, I have midterm exams !" said the other, as she ran out of coffee to refill her coffee with her water. "Well, they
Hello, I have midterm exams  and you have a lot of practice and time to practice.  You all have to follow your plan to get
Hello, I have midterm exams !"

That kind of thing, though, is not typical of a professional in this field.

And
Hello, I have midterm exams !"

"Of course. If I'm not too busy I'll take them tomorrow night. How you're"""

"""random model
DELL@MacBook-Pro week10 % python3 train_gpt2.py
Hello, I have midterm exams  Polkrakassertapan conflicting Guns Plus clear convicted Theorysafety theme Colo DualGames BordAGsimilar frustrost645 pictputing
Hello, I have midterm exams  KislyakTech Oversuders swarminnerkrit Mann condemnationLODDISTR Hussain decor ebook Taj reiterate chromos arrangingrobEurope strong�com
Hello, I have midterm exams  Oxford chair Ending theology Sarasalon mLoreAndOnlineArabildoaud0200 Productapes shrew Seen anguishrored356darkyon Exactly Nay
Hello, I have midterm exams  Authorization Yen fringe upside swarm puritycens Specialistorbit donor hidden JanetIss Heritage evaluations democracy FOIA Aston Allen aestIcestrength CPUs
Hello, I have midterm exams  similarities deserted congen Atl409 Gaia Edmonton authorities imposes explos Volunte SR interactionsIce Powered espresso compromputing omittedxb Murder Leafs minced
"""

"""
Merhaba, Şule ile İstanbul'a gidiyorum ȝo rajı, İstanbulğ
Merhaba, Şule ile İstanbul'a gidiyorum ile ösokampe İstanbul�
Merhaba, Şule ile İstanbul'a gidiyorum ışikları, segino yı
Merhaba, Şule ile İstanbul'a gidiyorum ile Şelta gidiyorum kü
Merhaba, Şule ile İstanbul'a gidiyorum ín bizah al-dun al-"""

"""step 49, loss: 0.0027097740676254034"""

"""gpt init: step 49, loss: 6.713193893432617"""

"""bfloat16 -> step 48, loss: 6.94805908203125, dt: 113.12ms, tok/sec: 1131.56
step 49, loss: 6.83148193359375, dt: 113.14ms, tok/sec: 1131.30
----------------------------------------------------------------------
steps                : 50
total wall time      : 5.90 s
first step           :  318.27 ms  (includes warmup)
steady mean          :  113.93 ms  (steps 5+)
steady median        :  113.43 ms
fastest / slowest    :  111.95 /  118.59 ms
throughput           :   1128.4 tok/sec  (from median)
----------------------------------------------------------------------
"""
"""torch compile
step 49, loss: 6.84906005859375, dt: 135.14ms, tok/sec: 947.19
----------------------------------------------------------------------
steps                : 50
total wall time      : 7.71 s
first step           :  916.13 ms  (includes warmup)
steady mean          :  139.15 ms  (steps 5+)
steady median        :  134.97 ms
fastest / slowest    :  126.57 /  152.27 ms
throughput           :    948.4 tok/sec  (from median)
----------------------------------------------------------------------"""
"""change vocab size
step 49, loss: 6.81964111328125, dt: 122.00ms, tok/sec: 1049.18
----------------------------------------------------------------------
steps                : 50
total wall time      : 6.03 s
first step           :  333.57 ms  (includes warmup)
steady mean          :  116.18 ms  (steps 5+)
steady median        :  115.02 ms
fastest / slowest    :  112.88 /  127.06 ms
throughput           :   1112.9 tok/sec  (from median)
----------------------------------------------------------------------
"""
#results above was using B=4, T=32, below using B=32 T=128
"""vocab size 50304
step 49, loss: 5.886235237121582, norm: 0.9397, dt: 4090.02ms, tok/sec: 8011.69
----------------------------------------------------------------------
steps                : 50
total wall time      : 206.33 s
first step           : 4114.43 ms  (includes warmup)
steady mean          : 4128.77 ms  (steps 5+)
steady median        : 4123.10 ms
fastest / slowest    : 4049.75 / 4233.49 ms
throughput           :   7947.4 tok/sec  (from median)
----------------------------------------------------------------------
"""
"""
step 49, loss: 5.9508771896362305, norm: 0.7897, dt: 4686.61ms, tok/sec: 6991.83
----------------------------------------------------------------------
steps                : 50
total wall time      : 237.60 s
first step           : 4708.94 ms  (includes warmup)
steady mean          : 4752.80 ms  (steps 5+)
steady median        : 4723.59 ms
fastest / slowest    : 4676.76 / 4906.13 ms
throughput           :   6937.1 tok/sec  (from median)
----------------------------------------------------------------------"""
"""
--- Why torch.compile did not help on this machine (Apple M5, MPS, torch 2.8) ---

Measured, bf16 autocast, median step time (eager vs torch.compile):
    B=4,  T=32    119.21 ms  ->  134.67 ms   (1.13x SLOWER)
    B=16, T=256   727.85 ms  -> 1233.51 ms   (1.69x SLOWER)

So it is not "the model is too small": the penalty grows with size. Reasons,
from reading Inductor's generated code:

1. No Metal matmul codegen. Every matmul is handed straight back to the same
   MPS kernels eager already uses (extern_kernels.mm x37, extern_kernels.bmm
   x72). The dominant cost of a step is therefore completely untouched.
2. The elementwise/softmax ops get replaced by many small generated Metal
   shaders (300+ mps_lib_* dispatches). These lose to MPSGraph's own fusion
   in eager, so the parts compile *does* change get slower.
3. "Not enough SMs to use max_autotune_gemm mode" is cosmetic, not the cause:
   torch/_inductor/utils.py gates GEMM autotuning on multi_processor_count >= 68
   (an RTX 3080). MPS reports 8, so the gate fails - but there is no Metal GEMM
   template to autotune anyway, per point 1. SMs are an NVIDIA concept.
4. It breaks on dynamic shapes: feeding two different (B,T) shapes in one
   process raises
       InductorError: TypeError: cannot determine truth value of Relational: s75 <= 1024
   from torch/_inductor/codegen/mps.py.

torch.compile's real wins (kernel fusion that avoids memory round-trips, CUDA
graphs to kill launch overhead) have no MPS equivalent in 2.8.

Also tested and rejected: F.scaled_dot_product_attention ("flash attention").
It is consistently a bit SLOWER here than the manual q @ k.T attention
(693 -> 744 ms at B=16,T=256), because MPS has no fused flash kernel to fall
into - that speedup is a CUDA result.

"""
"""final version
step 49, loss: 5.998026, lr: 3.0416e-05, norm: 0.7807, dt: 4853.94ms, tok/sec: 6750.80
----------------------------------------------------------------------
steps                : 50
total wall time      : 236.91 s
first step           : 4911.12 ms  (includes warmup)
steady mean          : 4732.30 ms  (steps 5+)
steady median        : 4698.66 ms
fastest / slowest    : 4623.90 / 5002.00 ms
throughput           :    871.7 tok/sec  (from median)
----------------------------------------------------------------------"""
"""200 step
step 199, loss: 5.503137, lr: 3.0000e-05, norm: 0.9102, dt: 4737.95ms, tok/sec: 6916.07
----------------------------------------------------------------------
steps                : 200
total wall time      : 960.73 s
first step           : 5058.24 ms  (includes warmup)
steady mean          : 4803.07 ms  (steps 5+)
steady median        : 4773.11 ms
fastest / slowest    : 4635.91 / 5256.04 ms
throughput           :    858.1 tok/sec  (from median)
----------------------------------------------------------------------
DELL@MacBook-Pro week11 % """