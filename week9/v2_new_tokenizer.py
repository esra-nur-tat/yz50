import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
dropout = 0.2
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
num_heads = 6
num_layers = 6

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"using device: {device}")
eval_iters = 200
n_embd = 384
 
torch.manual_seed(1337)
with open('turkce_egitim_1mb.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# --- BPE tokenizer (trained on the corpus, from tokenization.ipynb) ---
vocab_size = 512                 # 256 base byte tokens + learned merges
num_merges = vocab_size - 256

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# train the BPE merges on the raw UTF-8 bytes of the corpus
tokens = list(text.encode("utf-8"))
ids = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx
print(f"tokenizer trained: {len(tokens)} bytes -> {len(ids)} tokens "
      f"(compression {len(tokens)/len(ids):.2f}X), vocab_size={vocab_size}")

# build the vocab used for decoding
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def encode(s):
    toks = list(s.encode("utf-8"))
    while len(toks) >= 2:
        stats = get_stats(toks)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        toks = merge(toks, pair, merges[pair])
    return toks

def decode(l):
    toks = b"".join(vocab[idx] for idx in l)
    return toks.decode("utf-8", errors="replace")

#train val splits
data = torch.tensor(ids, dtype=torch.long)   # corpus already BPE-encoded above
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() #we don't need gradients for evaluation
def estimate_loss(): #to prevenet luck based losses, we will take average of 200 iterations
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)

        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,C)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation. communication is done by self-attention, computation is done by feedforward neural networks """

    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_head = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads=num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx=context, max_new_tokens=500)[0].tolist()))

""" first results of the bigram model are as follows:

step 2700: train loss 2.4738, val loss 2.4911

MARI he avayokis erceller thour d, myono thishe me tord se by he me, Forder anen: at trselorinjulour t yoru thrd wo ththathy IUShe bavidelanoby man ond be jus as g e atot Meste hrle s, ppat t JLENCOLIUS:
Oppid tes d s o ged moer y pevehear soue maramapay fo t: bueyo malalyo!
Duir.
Fl ke it I t l o'ddre d ondu s?
cr, havetrathackes w.
PUpee meshancun, hrendspouthoulouren whel's'sesoread pe, s whure our heredinsethes; sedsend r lo pamit,
QUMIVIVIOfe m ne RDINid we tr ort; t:
MINENXI l dintandore r"""

"""self attention model results are as follows:

step 4800: train loss 2.3838, val loss 2.4043

ARI
HAThan, thivet'ly meirar ysuch, fo thest
K
CAV:
HEDIERESTAD Pfr:
BENENENLING CENNUCINA IO: yond houstoruloubleles, besordweloe yot bed thed Ond Doue bou
His, to I lllat;
Kind yo Icad imoralceng, wpatsl my peer furs bust hasthe sor merals of ss whacer gaers, ver: biolllay seirost fat:
Thones som, wer ran
Toupls om?
Sll: hy athithos benciet bs fof.
Amathalcbenou liles caly, detave shpreearg
Whe wad wolf sd it totosthe, wank th in thit shin
S ht ande conit iul cert gpapam oun llkeavilint hmy r

"""
""" multihead attention results are as follows:
step 4800: train loss 2.2517, val loss 2.2729

bacont ya hom--wit'll meve renmuch,
I headst
K
Cond;
Kusbend rea'd ther sot hitl qiularsai thrutend houstorullsblelts, be sornw-loe you bed't that.

Dove bor
His for I 
That;
wind you cas hougaiceng, wpatsl my prow En's bust hasthe sor mes
ECWo has whaver gacps, var: biolllay shir sold apoe hats sont wer ran
To pos'l Heall, hady btth sheencies beres, me athe cin nos, yestis tand tant shpr, mig
Whe wad wolf slill tot sthe, wand ther thein shin enes and on: I wiul cort gor avesun lok, thrins hatne"""

"""feedForward added results:

step 4800: train loss 2.2106, val loss 2.2289

As you'stl orne'ns anden yerth at, shy her hell,
Anpar hape.

I I a is buinkere den whes for hind fad mod:
Whe thomiveord Wat, your.

PLAGOUCHALOLLost thy; to sond, oloy be thy
En arlie he semoher doer he of men.
Willadell.

KICICIORMOES:
Word grin:
Why and.

Tas imh cone you the gostestere, im, sousy thathy heerente leve slestroof and sker to serlonde, fin dand cond Civell wean your seath is your hom anels selight pand.
Farse andint
Gerins then't athwmest,
Afinghen.

NrOGLO:
Pon of everendeme p
"""
"""residual connection added results:
step 4800: train loss 1.9851, val loss 2.0778

DUKE VINCENTER:
Had the for and nrowning it is a ghich she swear, from 'date of to werking than EDWASS F ars is a by her ove in ercentlemanine spell'd planoble hope,
Srows way beck noble that, ox that knot of the so that, Heave to of his hook in him led proplaces pry me alough. But thesed end I nesturpeen'd;
Will faluew'd the thit stucued ut tapurs of names:
When mucch'd t ye,
The revil, Bonges!

MENVRKE VINCENCENTES:
Had womss they your in his good, make;
And there ter preak,
Farto miniss at p"""

"""layer normalization added results:
step 4800: train loss 1.9756, val loss 2.0682

LUCDINGS:
We, loid pingan.

POUTE:
Whing it is a not with me way, fro f'd on of the frue comes? EDWAPOR:
The is a bert;
How Richerc out lizens ap'd bord,
Do,
Belot as reas.

IVIOSCA:
A till man the geend thou the so that, Heave to of his too singer's thep hill me pry me allwic zevit ppeated your sir:
Ten's thring his with chath, betuh wawn trapurs of nastormine, ours suke ye,
Eke rether for of conty pas;
Thise to
talk I thansp taty your in his do you face the he revego peray
Menstor it as a sp
"""
"""multihead attention with feedforward and residual connection and layer normalization results:
step 4500: train loss 1.1071, val loss 1.4725

ESCALUS:
Enend, charity thee; I thither speak for off allswer enr.

First Senator:
A wise drop'sy, the treason is counted by unined,
Drepess abour.

CAMILLO:
Your judgmention since or memp reap night.

COMIPHENES:
We shall be so; I have speak and by me.

FLOUCIO:
O, if you must be holy pardon to-morrow hath,
That the heads of your honour hour.

FRIAR LAURENCE:
So, that in black ropeness with him: he's a hot;
By this carelet-kinsman out.

CLARENCE:

QUEEN:
From succeness are you:
Now with disconc"""

"""full model with Turkish text results:
step 4500: train loss 0.9062, val loss 1.2471

Padi gerilen —Eylül 35. Kolordu Kumandanlık mekteptedir.

Müstaceldir Kâzım

Ağa) Konya’da Müdafaa-i Hukuk Cemiyeti’nin Mustafa Kemal

Ali Kemal Beyefendi’ye

Sivas’tan, 20.9.30.35

Konya’dan dörderdiğim muhaberen dönmek için Ferit Paşa Kabinesi’nde, Temmuz Valisi Mahrir’de Erzurum Van Valisi de, Cemal Paşa’ya, kendine geldikçe ile mürâcaat ettik. Yeni Kabine, ictihâb-ı hakkın, fikriye, kendisini ve azm-i nokta-i nezarla olarak Cemal Paşa, bu Kongre’nin aki:

Telgrafını teshîdinde kat’î, vâki ku
"""