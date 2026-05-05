import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import re

# ----------------------------
# Inställningar
# ----------------------------
batch_size = 4
block_size = 64
max_iters = 10000
learning_rate = 3e-4

n_embd = 128
n_heads = 4
n_layers = 2

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ----------------------------
# Tokenizer
# ----------------------------
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for i, w in enumerate(vocab)}

    def tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.stoi[w] for w in tokens if w in self.stoi]

    def decode(self, indices):
        return " ".join([self.itos[i] for i in indices if i in self.itos])

# ----------------------------
# Läs data
# ----------------------------
with open("data_qa.txt", "r", encoding="utf-8") as f:
    text = f.read()

# skapa vocab med tokenizer
tokens = re.findall(r"\w+", text.lower())
vocab = sorted(list(set(tokens)))
vocab_size = len(vocab)

# spara vocab
with open("vocab_qa.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

tokenizer = Tokenizer(vocab)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# ----------------------------
# Batch
# ----------------------------
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ----------------------------
# Modell
# ----------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# ----------------------------
# Init modell
# ----------------------------
model = MiniTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ----------------------------
# Träning
# ----------------------------
print("Startar träning...")

for i in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        print(f"Step {i}, loss: {loss.item():.4f}")

# ----------------------------
# Spara
# ----------------------------
torch.save(model.state_dict(), "model_qa.pth")
print("Modell sparad")