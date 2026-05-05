from flask import Flask, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re

app = Flask(__name__)

# ----------------------------
# Inställningar
# ----------------------------
block_size = 64
n_embd = 128
n_heads = 4
n_layers = 2

temperature = 0.2
top_k = 5
max_new_tokens = 20

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
# Ladda vocab
# ----------------------------
with open("vocab_qa.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

tokenizer = Tokenizer(vocab)
vocab_size = len(vocab)

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
# Flask route
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    svar = ""

    if request.method == "POST":
        fråga = request.form["fråga"]

        prompt = f"Fråga: {fråga}\nSvar:"
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

        generated = model.generate(context)
        text = tokenizer.decode(generated[0].tolist())

        # Plocka ut bara svaret
        clean_answer = text

        if "svar" in clean_answer:
            clean_answer = clean_answer.split("svar", 1)[1]

        if "slut" in clean_answer:
            clean_answer = clean_answer.split("slut", 1)[0]

        svar = clean_answer.strip()

        if len(svar) > 0:
            svar = svar[0].upper() + svar[1:]

        with open("user_data.txt", "a", encoding="utf-8") as f:
            f.write(f"Fråga: {fråga}\n")
            f.write(f"Svar: {svar}\n")
            f.write("Slut.\n\n")

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Mini-LLM</title>
</head>
<body style="font-family: Arial; text-align:center; margin-top:100px;">
    <h1>Mini-LLM</h1>

    <form method="post">
        <input name="fråga" style="width:300px; padding:10px;" placeholder="Ställ en fråga..." />
        <button type="submit">Fråga</button>
    </form>

    <p style="margin-top:20px; font-size:18px;">{svar}</p>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5001)