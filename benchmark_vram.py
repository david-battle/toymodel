#!/usr/bin/env python3
"""VRAM benchmark for different model configs."""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import time
import gc

def make_model(n_layer, n_head, n_embd, block_size, vocab_size=50304):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(n_embd, bias=False)
            self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True, bias=False)
            self.ln2 = nn.LayerNorm(n_embd, bias=False)
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd, bias=False),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd, bias=False),
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
            self.tok = nn.Embedding(vocab_size, n_embd)
            self.pos = nn.Embedding(block_size, n_embd)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd, bias=False)
            self.head = nn.Linear(n_embd, vocab_size, bias=False)
            self.head.weight = self.tok.weight  # tied embeddings

        def forward(self, idx):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.tok(idx) + self.pos(pos)
            mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)
            for blk in self.blocks:
                x = blk(x, mask)
            return self.head(self.ln_f(x))

    return MiniGPT()

def benchmark_config(n_layer, n_head, n_embd, block_size, micro_batch, accum_steps, steps=5):
    torch.cuda.empty_cache()
    gc.collect()
    
    model = make_model(n_layer, n_head, n_embd, block_size).cuda()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n=== {param_count/1e6:.1f}M params | L{n_layer} H{n_head} D{n_embd} | ctx={block_size} | µ{micro_batch} x{accum_steps} ===")
    
    try:
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        
        for step in range(steps):
            x = torch.randint(0, 50304, (micro_batch, block_size), device='cuda')
            y = torch.randint(0, 50304, (micro_batch, block_size), device='cuda')
            
            for micro_step in range(accum_steps):
                with autocast(dtype=torch.float16):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits.view(-1, 50304), y.view(-1))
                    loss = loss / accum_steps
                scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        tok_per_sec = (micro_batch * block_size * accum_steps * steps) / elapsed
        
        print(f"  {steps} updates in {elapsed:.1f}s = {tok_per_sec:,.0f} tok/s")
        print(f"  peak alloc: {peak_alloc:.2f} GB | reserved: {peak_reserved:.2f} GB")
        print(f"  OK")
        return True, peak_alloc, tok_per_sec
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM: {e}")
        return False, None, None
    finally:
        del model, optimizer, scaler
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    configs = [
        # (n_layer, n_head, n_embd, block_size, micro_batch, accum_steps)
        # 85M configs
        (10, 10, 640, 512, 4, 64),
        (10, 10, 640, 512, 8, 32),
        (10, 10, 640, 512, 2, 128),
        # 125M configs
        (12, 12, 768, 512, 4, 64),
        (12, 12, 768, 512, 2, 128),
        (12, 12, 768, 512, 1, 256),
    ]
    
    results = []
    for cfg in configs:
        ok, peak, tok_s = benchmark_config(*cfg)
        results.append((cfg, ok, peak, tok_s))
        if not ok:
            print(f"  Stopping at first OOM for this model size")
            break
    
    print("\n=== SUMMARY ===")
    for (cfg, ok, peak, tok_s) in results:
        n_layer, n_head, n_embd, block_size, mb, acc = cfg
        params = 12 * n_layer * n_embd**2 + 50304 * n_embd
        status = f"{peak:.2f} GB, {tok_s:,.0f} tok/s" if ok else "OOM"
        print(f"  L{n_layer} H{n_head} D{n_embd} ({params/1e6:.0f}M) ctx={block_size} µ{mb}x{acc}: {status}")