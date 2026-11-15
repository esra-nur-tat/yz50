"""Character-level GPT for DNA sequences.

The architecture is the same as week 8's `v2.py`: token plus learned position
embeddings, `n_layer` pre-LN transformer blocks (causal self-attention and a
4x MLP), a final LayerNorm, and a linear head.

Two differences from week 8:

1. Attention uses `F.scaled_dot_product_attention(is_causal=True)` instead of
   the hand-written tril mask and softmax. The maths is identical; the fused
   kernel is faster on MPS. A hand-written fallback is kept for builds where
   SDPA is unavailable.
2. Week 8 scaled the attention logits by `n_embd ** -0.5`, but the standard
   formulation uses `head_size ** -0.5`. With six heads that made the logits
   a factor of 1/sqrt(6) too small, so attention was softer than intended.
   Both paths here use the correct scale.

Promoter sequences are independent 251 bp records rather than one continuous
genome, so they are not concatenated. Each sequence is prefixed with a BOS
token and batched as its own example. This keeps the model from learning
transitions across boundaries that do not exist, and it makes a per-sequence
loss well defined.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


def _sdpa_works():
    """Probe SDPA once. Some MPS builds raise; fall back to manual attention."""
    try:
        q = torch.zeros(1, 2, 4, 8)
        F.scaled_dot_product_attention(q, q, q, is_causal=True)
        return True
    except Exception:
        return False


SDPA_OK = _sdpa_works()


@dataclass
class DNAConfig:
    vocab_size: int = 6      # 0 = BOS, then the observed bases
    block_size: int = 251    # one promoter fits in the context
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    bos_id: int = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DNAConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.dropout_p = cfg.dropout
        self.use_sdpa = SDPA_OK
        if not SDPA_OK:
            self.register_buffer(
                "tril",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size)),
                persistent=False,
            )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)   # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Week 8's manual path, with the corrected scale: hd, not n_embd.
            wei = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.attn_dropout(wei)
            y = wei @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg: DNAConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: DNAConfig):
        super().__init__()
        self.sa_head = CausalSelfAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DNAGPT(nn.Module):
    def __init__(self, cfg: DNAConfig):
        super().__init__()
        self.config = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, return_hidden=False):
        """idx: (B, T) long. targets: (B, T) long or None."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"T={T} > block_size={self.config.block_size}"
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.reshape(B * T).long()
            )
        if return_hidden:
            return logits, loss, x
        return logits, loss

    # ---- helpers shared by training and analysis ----

    def make_xy(self, seqs):
        """seqs: (B, L) base ids without BOS -> x, y both (B, L).

        full = [BOS, b0, ..., b_{L-1}]   (length L + 1)
        x    = full[:L]  = [BOS, b0, ..., b_{L-2}]
        y    = full[1:]  = [b0,  b1, ..., b_{L-1}]

        Every base is predicted exactly once; the first one from BOS alone.
        """
        seqs = seqs.long()
        B, L = seqs.shape
        bos = torch.full((B, 1), self.config.bos_id, dtype=torch.long, device=seqs.device)
        full = torch.cat([bos, seqs], dim=1)
        return full[:, :L], full[:, 1:]

    @torch.no_grad()
    def per_position_nll(self, seqs):
        """seqs: (B, L) -> (B, L) negative log-likelihood in nats."""
        x, y = self.make_xy(seqs)
        B, L = y.shape
        logits, _ = self(x)
        nll = F.cross_entropy(
            logits.view(B * L, -1), y.reshape(B * L), reduction="none"
        )
        return nll.view(B, L)

    @torch.no_grad()
    def sequence_nll(self, seqs):
        """seqs: (B, L) -> (B,) mean NLL per base in nats."""
        return self.per_position_nll(seqs).mean(dim=1)

    @torch.no_grad()
    def embed(self, seqs):
        """seqs: (B, L) -> (B, n_embd) mean-pooled final hidden state."""
        x, _ = self.make_xy(seqs)
        _, _, h = self(x, return_hidden=True)
        return h.mean(dim=1)

    @torch.no_grad()
    def generate(self, n, length, device, temperature=1.0, forbid_bos=True,
                 prompt=None):
        """Sample sequences. -> (n, length) base ids, prompt included.

        prompt: (n, k) base ids to condition on, or None to start from BOS
        alone. With a prompt the first k columns of the result are the prompt
        itself and the model writes the remaining length - k bases, which is
        the same prompt-continuation setup as weeks 7 and 8.
        """
        idx = torch.full((n, 1), self.config.bos_id, dtype=torch.long, device=device)
        if prompt is not None:
            idx = torch.cat([idx, prompt.long().to(device)], dim=1)
        n_new = length - (idx.shape[1] - 1)
        for _ in range(n_new):
            cond = idx[:, -self.config.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / temperature
            if forbid_bos:
                logits[:, self.config.bos_id] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx[:, 1:]        # drop BOS


NATS_TO_BITS = 1.4426950408889634   # 1 / ln(2)
RANDOM_BITS_PER_BASE = 2.0          # uniform over A, C, G, T
