import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# ----------------------------
# Inställningar
# ----------------------------
block_size = 64
temperature = 0.3
top_k = 5
max_new_tokens = 30

n_embd = 128
n_heads = 4
n_layers = 2

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ----------------------------
# Läs vocab
# ----------------------------
with open("vocab_qa.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

vocab_size = len(vocab)

stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for i, w in enumerate(vocab)}

def encode(s):
    return [stoi[w] for w in s.lower().split() if w in stoi]

def decode(l):
    return " ".join([itos[i] for i in l])

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

    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        return self.lm_head(x)

    def generate(self, x):
        for _ in range(max_new_tokens):
            x_cond = x[:, -block_size:]
            logits = self(x_cond)

            logits = logits[:, -1, :]
            logits = logits / temperature

            values, indices = torch.topk(logits, top_k)
            probs = F.softmax(values, dim=-1)

            next_index = torch.multinomial(probs, num_samples=1)
            next_token = indices.gather(-1, next_index)

            x = torch.cat((x, next_token), dim=1)

        return x

# ----------------------------
# Ladda modell
# ----------------------------
model = MiniTransformer().to(device)
model.load_state_dict(torch.load("model_qa.pth", map_location=device))
model.eval()

# ----------------------------
# Generera
# ----------------------------
start = input("Skriv din fråga: ")

# AUTO FIX → lägger till format
if not start.lower().startswith("fråga:"):
    start = f"Fråga: {start} \nSvar:"

context = torch.tensor([encode(start)], dtype=torch.long).to(device)

generated = model.generate(context)

text = decode(generated[0].tolist())

# visa bara första svaret
if "slut." in text:
    text = text.split("slut.")[0] + "slut."

print("\n--- RESULTAT ---\n")
print(text)