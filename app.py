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
# Minne
# ----------------------------
last_topic = None

# ----------------------------
# Normalisering
# ----------------------------
def normalize_question(text):
    text = text.lower().strip()

    replacements = {
        "va ": "vad ",
        "va e ": "vad är ",
        "vad e ": "vad är ",
        "svergie": "sverige",
        "sveirge": "sverige",
        "stockhom": "stockholm",
        "stokholm": "stockholm",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"[?!]+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def update_topic(question):
    topics = ["sverige", "stockholm", "ai", "eu", "fotboll", "python", "internet"]
    for topic in topics:
        if topic in question:
            return topic
    return None

def apply_memory(question, topic):
    if topic is None:
        return question

    if question in ["vad är huvudstaden", "vad heter huvudstaden", "huvudstad"]:
        if topic == "sverige":
            return "vad är sveriges huvudstad"

    if question in ["hur många bor där", "befolkning"]:
        if topic == "sverige":
            return "hur många bor i sverige"

    return question

# ----------------------------
# Tokenizer
# ----------------------------
class Tokenizer:
    def __init__(self, vocab):
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for i, w in enumerate(vocab)}

    def tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def encode(self, text):
        return [self.stoi[w] for w in self.tokenize(text) if w in self.stoi]

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
# Ladda data (RAG)
# ----------------------------
qa_pairs = []

def load_qa():
    global qa_pairs
    qa_pairs = []

    with open("data_qa.txt", "r", encoding="utf-8") as f:
        content = f.read().lower()

    blocks = content.split("slut.")
    for block in blocks:
        if "fråga:" in block and "svar:" in block:
            try:
                q = block.split("fråga:")[1].split("svar:")[0].strip()
                a = block.split("svar:")[1].strip()
                qa_pairs.append((normalize_question(q), a))
            except:
                pass

load_qa()

# ----------------------------
# Similarity (bättre)
# ----------------------------
STOPWORDS = {
    "vad", "är", "en", "ett", "i", "på", "och", "som",
    "det", "den", "de", "kan", "förklara", "kort", "enkelt"
}

def important_words(text):
    words = re.findall(r"\w+", text.lower())
    return {w for w in words if w not in STOPWORDS}

def similarity(a, b):
    a_words = important_words(a)
    b_words = important_words(b)

    if len(a_words) == 0 or len(b_words) == 0:
        return 0

    return len(a_words & b_words) / len(a_words | b_words)

def find_best_match(user_question):
    best_score = 0
    best_answer = None

    for q, a in qa_pairs:
        score = similarity(user_question, q)

        if score > best_score:
            best_score = score
            best_answer = a

    return best_score, best_answer

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
    global last_topic

    svar = ""
    källa = ""

    if request.method == "POST":
        fråga_original = request.form["fråga"]
        fråga = normalize_question(fråga_original)

        new_topic = update_topic(fråga)
        if new_topic is not None:
            last_topic = new_topic

        fråga = apply_memory(fråga, last_topic)

        score, answer = find_best_match(fråga)

        # 🔥 RAG
        if score >= 0.75:
            svar = answer.strip()
            källa = "📚 Data"

        # 🤖 AI fallback
        else:
            prompt = f"Fråga: {fråga}\nSvar:"
            encoded = tokenizer.encode(prompt)

            if len(encoded) == 0:
                svar = "Jag förstod inte frågan."
                källa = "🤖 AI"
            else:
                context = torch.tensor([encoded], dtype=torch.long).to(device)
                generated = model.generate(context)
                text = tokenizer.decode(generated[0].tolist())

                if "svar" in text:
                    text = text.split("svar", 1)[1]

                if "slut" in text:
                    text = text.split("slut", 1)[0]

                svar = text.strip()
                källa = "🤖 AI"

        if len(svar) > 0:
            svar = svar[0].upper() + svar[1:]

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Mini-LLM</title>
</head>
<body style="font-family: Arial; text-align:center; margin-top:100px;">
    <h1>Mini-LLM</h1>

    <form method="post">
        <input name="fråga" style="width:350px; padding:10px;" placeholder="Ställ en fråga..." />
        <button type="submit">Fråga</button>
    </form>

    <p style="margin-top:20px; font-size:18px;">{svar}</p>
    <p style="font-size:14px; color:gray;">Källa: {källa}</p>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5001)