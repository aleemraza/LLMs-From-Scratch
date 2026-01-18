import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_blocks import Block

# ------------------ Device ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device:", device)

# ------------------ Dataset ------------------
dataset = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is aleem raza",
    "the roads of china are busy",
    "it is raining in china",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "eid is my favorite festival",
    "eid brings lights and sweets",
    "pakistan won the cricket match"
]

dataset = [s.lower() + " <end>" for s in dataset]
dataset = dataset * 50
random.shuffle(dataset)

text = " ".join(dataset)
tokens = text.split()

# ------------------ Vocabulary ------------------
words = []
for w in tokens:
    if w not in words:
        words.append(w)

vocab_size = len(words)
print("Vocab size:", vocab_size)

word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}

data = torch.tensor([word2idx[w] for w in tokens], dtype=torch.long)

# ------------------ Hyperparameters ------------------
block_size = 8
embedding_dim = 64
n_heads = 4
n_layers = 3
lr = 3e-4
epochs = 3000
batch_size = 32

# ------------------ Batch ------------------
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# ------------------ GPT Model ------------------
class SmallGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=5):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# ------------------ Init Weights ------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

model = SmallGPT().to(device)
model.apply(init_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ------------------ Training ------------------
for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 300 == 0:
        print(f"Step {step}/{epochs} | Loss: {loss.item():.4f}")

# ------------------ Generation ------------------
context = torch.tensor([[word2idx["aleem"]]], device=device)
out = model.generate(context, max_new_tokens=5)

print("\nGenerated text:\n")
print(" ".join(idx2word[int(i)] for i in out[0]))
