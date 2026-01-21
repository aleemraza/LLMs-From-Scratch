import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import os
from transformer_blocks import Block

# ------------------ Reproducibility ------------------
torch.manual_seed(42)

# ------------------ Device ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device:", device)

# ------------------ Load Dataset ------------------
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ------------------ Train SentencePiece (only once) ------------------
if not os.path.exists("tokenizer.model"):
    spm.SentencePieceTrainer.Train(
        input="dataset.txt",
        model_prefix="tokenizer",
        vocab_size=40,
        model_type="bpe",
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3
    )

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

ids = sp.encode(text, out_type=int)
data = torch.tensor(ids, dtype=torch.long)

vocab_size = sp.get_piece_size()
print("Vocab size:", vocab_size)

# ------------------ Hyperparameters ------------------
block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500
batch_size = 32

# ------------------ Batch Function ------------------
def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
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
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=10):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx

# ------------------ Init Model ------------------
model = SmallGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ------------------ Training ------------------
model.train()
for step in range(epochs):
    xb, yb = get_batch()
    _, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")

# ------------------ Generation ------------------
context = torch.tensor(
    [sp.encode("hello", out_type=int)],
    dtype=torch.long
).to(device)

out = model.generate(context, max_new_tokens=15, temperature=0.8, top_k=8)

generated_ids = out[0].tolist()
print("\nGenerated text:\n")
print(sp.decode(generated_ids))
