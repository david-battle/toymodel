"""Toy-model training loop (nanoGPT-style) for the ~51M/124M-param GPT.

Loads the finished tokenized corpus (see prepare_data.py), trains with the
60/28/12 arXiv/SE/Wikipedia mixture, fp16 AMP + GradScaler, AdamW with
weight-decay grouping, a token-counted cosine LR schedule, periodic and
signal-driven atomic checkpointing with suspend/resume.

The data loader is a simple synchronous document sampler (no worker
prefetch queues) whose RNG state is saved in each checkpoint, so resume
is bit-exact in the data sequence.

Usage:
  .venv/bin/python train.py                      # pilot, 20M tokens (51M model)
  .venv/bin/python train.py --budget 1000000000  # full run target
  .venv/bin/python train.py --resume ckpt/last.pt
  .venv/bin/python train.py --n-layer 12 --n-head 12 --n-embd 768 --block 512 --budget 2500000000  # 124M model, 2.5B tokens

Ctrl-C / SIGTERM sets a flag; the loop checkpoints and exits at the next
complete update. A second Ctrl-C forces immediate exit without saving.
"""

import argparse
import hashlib
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.resolve()
TOKENS = ROOT / "corpus" / "tokens"
CKPT_DIR = ROOT / "ckpt"

DEV = "cuda"
VOCAB = 50304
VOCAB_REAL = 50257

# Default model config (51M model) — can be overridden via CLI
N_LAYER = 8
N_HEAD = 8
N_EMBD = 512
BLOCK = 256

# Final recorded mixture (see plan 3): arXiv 60% / SE 28% / Wikipedia 12%.
# The SE share is subdivided among its sub-sources proportional to tokens.
MIX = {"arxiv": 0.60, "wikipedia": 0.12, "SE": 0.28}
SE_SOURCES = ["se-math", "se-physics", "se-stats", "se-chemistry", "se-cstheory"]

PEAK_LR = 6e-4
MIN_LR = 6e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EFF_TOKENS = 128 * 1024  # effective tokens per update (via accumulation)

# Runtime config (set in main)
CFG_N_LAYER = N_LAYER
CFG_N_HEAD = N_HEAD
CFG_N_EMBD = N_EMBD
CFG_BLOCK = BLOCK


# ---------------------------------------------------------------------------
# model (same recipe as gpu_smoke.py / benchmark.py)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
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
    def __init__(self, n_layer, n_head, n_embd, block_size, vocab_size=VOCAB):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=False)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings
        self.block_size = block_size
        self.n_layer = n_layer

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))


def make_model(n_layer, n_head, n_embd, block_size):
    model = MiniGPT(n_layer, n_head, n_embd, block_size).to(DEV)
    # nanoGPT-style init: std 0.02 everywhere, residual output projections
    # scaled by 1/sqrt(2*n_layer) so the residual stream stays ~O(1).
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, 0.0, 0.02)
    proj_std = 0.02 / (2 * n_layer) ** 0.5
    for blk in model.blocks:
        nn.init.normal_(blk.attn.out_proj.weight, 0.0, proj_std)
        nn.init.normal_(blk.mlp[-1].weight, 0.0, proj_std)
    return model


def get_block():
    return CFG_BLOCK


def decay_param_groups(model):
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.dim() < 2 else decay).append(p)
    return [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class LossFn(nn.Module):
    """Cross-entropy masking padded-vocab IDs so unused tokens can't win."""

    def __init__(self):
        super().__init__()
        self.ignore_index = -100

    def forward(self, logits, target):
        B, T, V = logits.shape
        t = target.clone()
        t[t >= VOCAB_REAL] = self.ignore_index
        return nn.functional.cross_entropy(
            logits.view(B * T, V), t.view(B * T), ignore_index=self.ignore_index
        )


# ---------------------------------------------------------------------------
# document sampler over the tokenized corpus
# ---------------------------------------------------------------------------
# sources: list of dicts {name, mmap (uint16 memmap), offsets (int64 array)}
# The sampler picks sources at the recorded mix, then a random document and a
# random window within it, filling BLOCK tokens (cyclic within the doc).

def load_sources():
    sources = []
    se_tokens = {}
    for tok_file in sorted(TOKENS.glob("*.bin")):
        name = tok_file.stem
        off = np.load(TOKENS / f"{name}.offsets.npy")
        mmap = np.memmap(tok_file, dtype=np.uint16, mode="r")
        s = {"name": name, "mmap": mmap, "offsets": off,
             "weight": MIX.get(name, 0.0)}
        sources.append(s)
        if name in SE_SOURCES:
            se_tokens[name] = int(off[-1])
    # subdivide the SE share proportionally to tokens
    se_total = sum(se_tokens.values())
    if se_total > 0:
        for s in sources:
            if s["name"] in SE_SOURCES:
                s["weight"] = MIX["SE"] * se_tokens[s["name"]] / se_total
    return [s for s in sources if s["weight"] > 0]


def _normalise_weights(sources):
    w = [s.get("weight", 0.0) for s in sources]
    tot = sum(w)
    return [x / tot for x in w]


def corpus_hash(sources):
    h = hashlib.sha256()
    for s in sorted(sources, key=lambda x: x["name"]):
        h.update(s["name"].encode())
        h.update(str(len(s["offsets"])).encode())
        h.update(str(int(s["offsets"][-1])).encode())
        h.update(str(Path(str(TOKENS / f"{s['name']}.bin")).stat().st_size).encode())
    return h.hexdigest()[:16]


def sample_batch(sources, weights, gen, b, device):
    """Draw a B x BLOCK token batch, one document window per row."""
    B = b
    block = get_block()
    cats = torch.multinomial(torch.tensor(weights, device="cpu"), B, replacement=True,
                             generator=gen)
    idx = torch.empty((B, block), dtype=torch.long)
    for i, ci in enumerate(cats.tolist()):
        src = sources[ci]
        off = src["offsets"]
        ndocs = len(off) - 1
        di = int(torch.randint(0, ndocs, (1,), generator=gen).item())
        start, end = int(off[di]), int(off[di + 1])
        doc = src["mmap"][start:end]
        dlen = doc.shape[0]
        if dlen >= block:
            s = int(torch.randint(0, dlen - block + 1, (1,), generator=gen).item())
            row = doc[s:s + block]
        else:
            reps = (block + dlen - 1) // dlen
            row = np.resize(doc, reps * dlen)[:block]
        idx[i] = torch.from_numpy(row.astype(np.int64))
    return idx.to(device)


class Sampler:
    """Holds the RNG used for both source draw and window draw, so saving the
    generator state makes the data sequence bit-repeatable on resume."""

    def __init__(self, sources):
        self.sources = sources
        self.weights = _normalise_weights(sources)
        self.gen = torch.Generator()

    def batch(self, b, device):
        return sample_batch(self.sources, self.weights, self.gen, b, device)


# ---------------------------------------------------------------------------
# LR schedule in tokens
# ---------------------------------------------------------------------------

def lr_at(tokens, budget):
    warm = int(0.02 * budget)
    if tokens <= warm:
        return PEAK_LR
    frac = (tokens - warm) / max(1, budget - warm)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * frac))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def _atomic_save(state, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(path)
    try:
        dirfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except OSError:
        pass


def save_checkpoint(model, opt, scaler, sampler, step, tokens_seen, best_val,
                    config, why):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "fmt": 1,
        "config": config,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "rng": sampler.gen.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(torch.device(DEV)),
        "time": time.time(),
        "why": why,
    }
    last = CKPT_DIR / "last.pt"
    _atomic_save(state, last)
    # rotate: keep a step-<N>.pt every 10k steps, only the most recent 3
    if step % 10000 == 0:
        snap = CKPT_DIR / f"step-{step}.pt"
        _atomic_save(state, snap)
        snaps = sorted(CKPT_DIR.glob("step-*.pt"))
        for old in snaps[:-3]:
            old.unlink(missing_ok=True)
    size = last.stat().st_size / 1e9
    print(f"[ckpt] saved last.pt ({size:.2f} GB) tokens={tokens_seen/1e6:.1f}M "
          f"step={step} why={why}", flush=True)


def load_checkpoint(path, model, opt, scaler, sampler, config):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["fmt"] != config["fmt"]:
        sys.exit(f"[resume] checkpoint format {state['fmt']} != {config['fmt']}")
    if state["config"]["corpus_hash"] != config["corpus_hash"]:
        sys.exit("[resume] corpus hash mismatch; corpus changed")
    for k in ("n_layer", "n_head", "n_embd", "block_size"):
        if state["config"].get(k) != config.get(k):
            sys.exit(f"[resume] model config mismatch: {k}={state['config'].get(k)} != {config.get(k)}")
    model.load_state_dict(state["model"])
    model = model.to(DEV)
    opt.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    sampler.gen.set_state(state["rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], torch.device(DEV))
    return state


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

_graceful = False
_force = False


def _handler(signum, frame):
    global _graceful, _force
    if _graceful:
        print("[sig] second signal: forcing immediate exit", flush=True)
        _force = True
        return
    _graceful = True
    print("\n[sig] graceful pause requested (will checkpoint at next update)",
          flush=True)


def install_signals():
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# evaluation on held-out documents
# ---------------------------------------------------------------------------

def build_eval_set(sampler, n_docs=64):
    """Hold out a fixed set of document windows per source for eval, drawn once
    at startup so evaluation does not consume training RNG state."""
    block = get_block()
    ev = []
    for s in sampler.sources:
        off = s["offsets"]
        ndocs = len(off) - 1
        g = torch.Generator()
        g.manual_seed(int(hashlib.sha256(s["name"].encode()).hexdigest()[:8], 16))
        keep = min(n_docs, ndocs)
        di = torch.randint(0, ndocs, (keep,), generator=g)
        rows = torch.empty((keep, block), dtype=torch.long)
        for j, d in enumerate(di.tolist()):
            start, end = int(off[d]), int(off[d + 1])
            doc = s["mmap"][start:end]
            dlen = doc.shape[0]
            if dlen >= block:
                st2 = int(torch.randint(0, dlen - block + 1, (1,), generator=g).item())
                row = doc[st2:st2 + block]
            else:
                reps = (block + dlen - 1) // dlen
                row = np.resize(doc, reps * dlen)[:block]
            rows[j] = torch.from_numpy(row.astype(np.int64))
        ev.append((s["name"], rows))
    return ev


def evaluate(model, lossf, eval_sets, tokens_seen):
    model.eval()
    per_source = {}
    with torch.no_grad():
        for name, rows in eval_sets:
            device_rows = rows.to(DEV)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(device_rows[:, :-1])
                loss = lossf(logits, device_rows[:, 1:].contiguous())
            per_source[name] = loss.item()
    model.train()
    se = [v for k, v in per_source.items() if k in SE_SOURCES]
    agg = (MIX["arxiv"] * per_source.get("arxiv", 0.0)
           + MIX["SE"] * (sum(se) / len(se) if se else 0.0)
           + MIX["wikipedia"] * per_source.get("wikipedia", 0.0))
    bits = "  ".join(f"{k}={v:.3f}" for k, v in per_source.items())
    print(f"[eval] tok={tokens_seen/1e6:.1f}M  agg={agg:.3f}  {bits}", flush=True)
    return agg


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global CFG_N_LAYER, CFG_N_HEAD, CFG_N_EMBD, CFG_BLOCK
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=None,
                    help="total tokens to consume (default 20M; on resume, "
                         "keeps the checkpoint's original budget unless "
                         "overridden — overriding starts a new schedule)")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="sequences per micro-batch")
    ap.add_argument("--accum", type=int, default=None,
                    help="gradient accumulation count (default = eff/micro)")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint to resume from")
    ap.add_argument("--ckpt-steps", type=int, default=500,
                    help="auto-checkpoint every N updates")
    ap.add_argument("--eval-steps", type=int, default=200,
                    help="run held-out evaluation every N updates")
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-accumulate", action="store_true")
    # Model config overrides
    ap.add_argument("--n-layer", type=int, default=N_LAYER)
    ap.add_argument("--n-head", type=int, default=N_HEAD)
    ap.add_argument("--n-embd", type=int, default=N_EMBD)
    ap.add_argument("--block", type=int, default=BLOCK)
    args = ap.parse_args()

    # Apply model config
    CFG_N_LAYER = args.n_layer
    CFG_N_HEAD = args.n_head
    CFG_N_EMBD = args.n_embd
    CFG_BLOCK = args.block

    user_budget = args.budget
    if args.budget is None:
        args.budget = 20_000_000

    install_signals()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_sources()
    for s in sources:
        print(f"  source {s['name']:<14} weight={s.get('weight', MIX.get(s['name'],0)):.3f} "
              f"docs={len(s['offsets'])-1} tokens={int(s['offsets'][-1])/1e6:.1f}M", flush=True)
    corphash = corpus_hash(sources)

    model = make_model(CFG_N_LAYER, CFG_N_HEAD, CFG_N_EMBD, CFG_BLOCK)
    diff = sum(p.numel() for p in model.parameters())
    print(f"params: {diff/1e6:.2f}M  block_size={CFG_BLOCK}  budget={args.budget/1e6:.0f}M", flush=True)

    scaler = torch.amp.GradScaler("cuda")
    opt = torch.optim.AdamW(decay_param_groups(model), lr=PEAK_LR,
                            betas=(0.9, 0.95), weight_decay=0.0)
    lossf = LossFn()

    sampler = Sampler(sources)

    micro = args.micro_batch * CFG_BLOCK
    if args.no_accumulate:
        accum = 1
    else:
        accum = args.accum or max(1, EFF_TOKENS // micro)
    eff = accum * micro
    print(f"micro-batch {args.micro_batch}x{CFG_BLOCK} = {micro} tok; "
          f"accum x{accum} -> {eff/1000:.0f}k tokens/update", flush=True)

    step = 0
    tokens_seen = 0
    best_val = float("inf")

    config = {
        "fmt": 1,
        "corpus_hash": corphash,
        "budget": args.budget,
        "micro_batch": args.micro_batch,
        "accum": accum,
        "seed": args.seed,
        "n_layer": CFG_N_LAYER,
        "n_head": CFG_N_HEAD,
        "n_embd": CFG_N_EMBD,
        "block_size": CFG_BLOCK,
    }

    if args.resume:
        if args.no_accumulate:
            sys.exit("--resume with --no-accumulate is not supported")
        st = load_checkpoint(args.resume, model, opt, scaler, sampler, config)
        step = st["step"]
        tokens_seen = st["tokens_seen"]
        best_val = st["best_val"]
        ckpt_budget = st["config"]["budget"]
        if user_budget is None:
            args.budget = ckpt_budget  # preserve the original schedule
        elif user_budget != ckpt_budget:
            print(f"[resume] NOTE: overriding budget {ckpt_budget} -> "
                  f"{user_budget}; this resets the LR schedule (new experiment)",
                  flush=True)
        print(f"[resume] step={step} tokens_seen={tokens_seen/1e6:.1f}M "
              f"best_val={best_val:.4f} budget={args.budget/1e6:.0f}M", flush=True)
    else:
        torch.manual_seed(args.seed)
    config["budget"] = args.budget

    eval_sets = build_eval_set(sampler)

    print(f"[train] starting tokens_seen={tokens_seen/1e6:.1f}M "
          f"lr={lr_at(tokens_seen,args.budget):.2e}", flush=True)

    last_t = time.time()
    last_tok = tokens_seen

    while tokens_seen < args.budget:
        opt.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for _ in range(accum):
            x = sampler.batch(args.micro_batch, DEV)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x[:, :-1])
                loss = lossf(logits, x[:, 1:].contiguous())
            micro_loss += loss.detach()
            scaler.scale(loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        skipped = 1 if scaler.step(opt) else 0
        scaler.update()
        step += 1
        tokens_seen += eff
        opt.param_groups[0]["lr"] = lr_at(tokens_seen, args.budget)
        opt.param_groups[1]["lr"] = lr_at(tokens_seen, args.budget)

        now = time.time()
        if step % args.log_steps == 0:
            dt = now - last_t
            rate = (tokens_seen - last_tok) / dt
            print(f"[step {step}] tok={tokens_seen/1e6:.2f}M "
                  f"loss={(micro_loss/accum).item():.3f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} "
                  f"scale={scaler.get_scale():.0f} skip={skipped} tok/s={rate:.0f}",
                  flush=True)
            last_t, last_tok = now, tokens_seen

        if step % args.eval_steps == 0:
            val = evaluate(model, lossf, eval_sets, tokens_seen)
            if val < best_val:
                best_val = val
                _atomic_save({
                    "fmt": config["fmt"], "config": config,
                    "model": model.state_dict(), "best_val": val,
                    "why": "best"}, CKPT_DIR / "best.pt")

        if step % args.ckpt_steps == 0:
            save_checkpoint(model, opt, scaler, sampler, step, tokens_seen,
                            best_val, config, "periodic")

        if _force:
            sys.exit(1)
        if _graceful:
            save_checkpoint(model, opt, scaler, sampler, step, tokens_seen,
                            best_val, config, "signal")
            print("[sig] paused; resume with --resume ckpt/last.pt", flush=True)
            sys.exit(0)

    save_checkpoint(model, opt, scaler, sampler, step, tokens_seen,
                    best_val, config, "budget-exhausted")
    print(f"[train] finished {tokens_seen/1e6:.1f}M tokens; best_val={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
