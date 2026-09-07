"""Sustained-throughput benchmark for the ~51M-param GPT on this GPU.

Runs the real training loop forward/backward/optimizer for a sustained period
(long enough to escape warmup and surface any throttling) and reports the
achieved tokens/sec, VRAM, and (best-effort) power/temperature. This feeds the
plan's batch/accumulation choices and the time estimate for a full run.

It mirrors the training recipe the plan intends for train.py as closely as a
short script can: fp16 AMP + GradScaler, AdamW, gradient accumulation, global
norm clipping, and padding-mask in the loss (IDs >= VOCAB_REAL). It uses
synthetic data, not the real corpus.

Usage:
  .venv/bin/python benchmark.py --seconds 30 --micro-batch 8
  .venv/bin/python benchmark.py --seconds 30 --micro-batch 8 16 32
"""

import argparse
import math
import subprocess
import time

import torch
import torch.nn as nn

torch.manual_seed(0)

DEV = "cuda"
VOCAB = 50304        # padded table size (masked IDs >= VOCAB_REAL are ignored)
VOCAB_REAL = 50257   # GPT-2 tokenizer vocabulary
N_LAYER = 8
N_HEAD = 8
N_EMBD = 512
BLOCK = 256

# Effective batch target (tokens/update), matching the plan's ~131k.
EFF_TOKENS = 128 * 1024


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


class LossFn(nn.Module):
    """Cross-entropy that masks padded-vocab IDs so unused tokens can't win."""

    def __init__(self, vocab_real):
        super().__init__()
        self.ignore_index = -100
        self.vocab_real = vocab_real

    def forward(self, logits, target):
        # Mark non-real vocab positions in the target as ignore so the model
        # is free to (and not penalized against) never emitting padded tokens.
        B, T, V = logits.shape
        t = target.clone()
        t[t >= self.vocab_real] = self.ignore_index
        return nn.functional.cross_entropy(
            logits.view(B * T, V), t.view(B * T), ignore_index=self.ignore_index
        )


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def smi(query):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--micro-batch", type=int, nargs="+", default=[8])
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--no-accumulate", action="store_true")
    args = ap.parse_args()

    model = MiniGPT().to(DEV)
    n = count_params(model)
    print(f"params: {n/1e6:.2f}M  block_size: {BLOCK}")

    for p in model.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, 0.0, 0.02)
    proj_std = 0.02 / (2 * N_LAYER) ** 0.5
    for blk in model.blocks:
        torch.nn.init.normal_(blk.attn.out_proj.weight, 0.0, proj_std)
        torch.nn.init.normal_(blk.mlp[-1].weight, 0.0, proj_std)

    scaler = torch.amp.GradScaler("cuda")
    opt = torch.optim.AdamW(
        model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    lossf = LossFn(VOCAB_REAL)
    target = torch.randint(0, VOCAB_REAL, (BLOCK,), device=DEV)

    base_pwr = smi("power.draw,memory.used")

    for micro_batch in args.micro_batch:
        B = micro_batch
        idx = torch.randint(0, VOCAB_REAL, (B, BLOCK), device=DEV)
        tgt_tile = target.unsqueeze(0).expand(B, -1).contiguous()

        tracked = {"nfinite": 0, "first": True, "loss": None}

        def step():
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(idx)
                    loss = lossf(logits, tgt_tile)
                if tracked["first"] and torch.isfinite(loss):
                    tracked["nfinite"] += 1
                scaler.scale(loss / accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tracked["first"] = False
            tracked["loss"] = loss

        eff = EFF_TOKENS if not args.no_accumulate else B * BLOCK
        accum = max(1, eff // (B * BLOCK))
        eff = accum * B * BLOCK
        micro_tok = B * BLOCK * accum

        print(f"\n--- micro-batch {B} x {BLOCK} = {B*BLOCK} tok; "
              f"accum x{accum} -> {eff/1000:.0f}k tok/update ---")

        for _ in range(args.warmup_steps):
            step()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        steps = 0
        min_pwr = None
        tracked["nfinite"] = 0
        tracked["first"] = True
        while time.perf_counter() - t0 < args.seconds:
            step()
            steps += 1
            p = smi("power.draw,memory.used")
            if p:
                parts = p.split(",")
                try:
                    w = float(parts[0].strip())
                    mi = float(parts[1].strip())
                    min_pwr = (w, mi) if min_pwr is None else (max(min_pwr[0], w), mi)
                except ValueError:
                    pass
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok = micro_tok * steps
        peak = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"  steps: {steps}  time: {dt:.1f}s")
        print(f"  tok/s: {tok/dt:.0f}  ms/update: {dt/steps*1000:.1f}")
        print(f"  loss (first update): {tracked['loss'].item():.4f}  "
              f"finite micro-losses: {tracked['nfinite']}/{accum}  "
              f"scale: {scaler.get_scale():.0f}")
        print(f"  peak alloc: {peak:.2f} GB  reserved: {reserved:.2f} GB")
        if min_pwr:
            print(f"  peak power: {min_pwr[0]:.0f} W  mem used: {min_pwr[1]:.0f} MiB")
        else:
            print("  (power/temp readout unavailable)")


if __name__ == "__main__":
    main()
