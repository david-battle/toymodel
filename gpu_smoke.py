"""CUDA toolchain smoke test.

Builds the plan's ~51M-param GPT shape and runs real fp16 AMP
forward/backward/AdamW steps on synthetic tokens. Verifies that the
torch/CUDA stack works on this GPU (loss starts near ln(vocab), decreases,
params stay finite) and prints a short-burst tokens/sec figure. It is a
toolchain check, not the training recipe: no weight-decay grouping, no
accumulation, no real data.

Run: .venv/bin/python gpu_smoke.py
"""

import math
import time

import torch
import torch.nn as nn

torch.manual_seed(0)

DEV = "cuda"
VOCAB = 50304
N_LAYER = 8
N_HEAD = 8
N_EMBD = 512
BLOCK = 256


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD, bias=False)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(N_EMBD, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD, bias=False),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD, bias=False),
        )

    def forward(self, x, mask):
        # pre-LayerNorm residual block (returns the full residual stream)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD, bias=False)
        self.head = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=DEV)
        x = self.tok(idx) + self.pos(pos)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=DEV), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    model = MiniGPT().to(DEV)
    n = count_params(model)
    print(f"params: {n/1e6:.2f}M")
    # nanoGPT-style init: std 0.02 everywhere, residual output projections
    # scaled by 1/sqrt(2*n_layer) so the residual stream stays ~O(1).
    for p in model.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, 0.0, 0.02)
    proj_std = 0.02 / (2 * N_LAYER) ** 0.5
    for blk in model.blocks:
        torch.nn.init.normal_(blk.attn.out_proj.weight, 0.0, proj_std)
        torch.nn.init.normal_(blk.mlp[-1].weight, 0.0, proj_std)

    scaler = torch.amp.GradScaler("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    lossf = nn.CrossEntropyLoss()

    # micro-batch: 8 seq x 256 = 2048 tokens
    B = 8
    T = BLOCK
    idx = torch.randint(0, VOCAB - 256, (B, T), device=DEV)
    target = torch.randint(0, VOCAB - 256, (B, T), device=DEV)

    # report loss before training
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss0 = lossf(model(idx).view(-1, VOCAB), target.view(-1)).item()
    print(f"initial loss: {loss0:.3f}  (uniform over vocab = {math.log(VOCAB):.2f})")

    # warmup a few steps
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(idx)
            loss = lossf(logits.view(-1, VOCAB), target.view(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nsteps = 20
    for _ in range(nsteps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(idx)
            loss = lossf(logits.view(-1, VOCAB), target.view(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tokens = B * T * nsteps
    print(f"final loss: {loss.item():.4f}  scale: {scaler.get_scale():.0f}")
    print(f"time: {dt:.2f}s  tok/s: {tokens/dt:.0f}  ms/step: {dt/nsteps*1000:.1f}")

    allfinite = all(torch.isfinite(v).all().item() for v in model.parameters())
    print(f"all params finite: {allfinite}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak alloc: {peak:.2f} GB")
    assert torch.isfinite(loss).all().item()
    assert allfinite


if __name__ == "__main__":
    main()