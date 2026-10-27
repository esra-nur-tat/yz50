from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math 

# Multi-head causal self-attention (diff vs week 8: q, k, v are computed in one single linear layer)
class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)  # one layer produces q, k and v together
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)    # output projection
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

model = GPT(GPTConfig())  # randomly initialized model
model.to('cpu')

# --- First training steps: overfit a single batch ---
# At random init the loss should be close to ln(50257) ~= 10.8,
# because the model is just guessing uniformly over all 50257 tokens.
# Training on the same batch many times should drive the loss toward 0;
# if it does, forward + backward are wired correctly.
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(50):
    optimizer.zero_grad()           # clear old gradients
    logits, loss = model(x, y)      # forward pass + loss
    loss.backward()                 # backprop: compute gradients
    optimizer.step()                # update the weights
    print(f"step {step}, loss: {loss.item()}")

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