# Toy Language Model — Full Plan

This document is the authoritative design for a small, GPU-trained English
language model focused on post-1900 STEM text. It covers the model
architecture, the curated corpus, the training plan (including suspend/resume),
and operational conventions. Companion file: `AGENTS.md` (current state +
working rules).

---

## 1. Hardware & environment (measured)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Super (Max-Q), **8 GB VRAM** |
| Driver / CUDA | Driver 566.03, CUDA 12.7 |
| Host | WSL2 (Linux 6.18 msft), ~27 GB RAM |
| Python | 3.14.4 |
| torch | 2.14.0 (installable; only release that supports 3.14) |
| Disk | ~936 GB free |

**GPU access note**: GPU is reachable from WSL2 via CUDA (the sibling
`raylib-test` repo documents WSLg/D3D12 graphics; for training we use CUDA
through torch). The card is power-capped at 80 W, so expect lower throughput
than a desktop 2080 Super.

---

## 2. Model design

A small decoder-only GPT, nanoGPT-style. Target ~50M params.

| Hyperparameter | Value |
|---|---|
| `n_layer` | 8 |
| `n_head` | 8 |
| `n_embd` | 768 |
| `block_size` (context) | 256 |
| `vocab_size` | ~50,000 (BPE via `tiktoken` cl100k_base, resized) |
| Params (approx) | ~50 M |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1) |
| Precision | fp16 mixed precision (AMP) |
| Batch | ~16 sequences × 256 tokens ≈ 4,096 tokens/step |
| LR | cosine schedule, peak ~3e-4, warmup ~500 steps |

**Memory model (fp16 + AdamW ≈ 18 bytes/param)**: 50M × 18 ≈ 0.9 GB for
weights + optimizer + grads. With activations, well within 8 GB at a ~4k-token
batch. Headroom exists to raise the batch size if benchmarking shows it.

**Why 256 context**: small model, small corpus; longer context dilutes the
limited weight capacity. 256 is enough to learn local coherence and basic
grammar.

---

## 3. Corpus (the 50/20/20/10 mix)

Lean heavily English, post-1900, STEM. All sources are openly licensed.

| Share | Source | Role |
|---|---|---|
| **50%** | **arXiv open-access papers** (physics, math, CS subsets) | STEM backbone |
| **20%** | **Feynman Lectures** (feynmanlectures.caltech.edu) | clean explanatory physics prose |
| **20%** | **English Wikipedia STEM articles** (physics/math/CS/chem categories) | broad modern vocabulary |
| **10%** | **StackExchange / Reddit STEM Q&A** (AskPhysics, StackOverflow, r/science) | conversational Q&A style |

### Curation goals
The user explicitly wants the *best* papers/articles/questions in the slice,
not a blind dump. Curation strategy:

1. **arXiv**: restrict to open-access (CC-BY) papers; drop non-STEM categories;
   filter out metadata/citation noise; prefer highly-cited or "seminal" papers
   where feasible. Light cleaning: strip LaTeX commands, references, author
   blocks.
2. **Wikipedia**: pull only STEM category articles (Category:Physics,
   Category:Mathematics, Category:Computer science, Category:Chemistry);
   strip markup/templates/infoboxes.
3. **Feynman**: clean, high-value prose; minimal cleaning.
4. **Q&A**: dedupe, drop junk/low-score posts (score threshold), keep the
   question + top accepted answer.

### Target size
- **Start (overnight validation)**: ~10–50 M tokens.
- **Full run**: ~1 B tokens (20 × 50 M params, Chinchilla-optimal).

Keep the raw slice separate from the tokenized training set so re-tokenizing
with a different vocab doesn't require re-downloading.

---

## 4. Tokenization

Use `tiktoken` `cl100k_base` (GPT-4 BPE). Resize/trim vocab to a smaller
tokenizer (`~50k`) for a lean embedding table — full size not needed for a toy.
Tokenize the curated corpus offline to `.bin`/`.npy` chunks (like nanoGPT's
`prepare.py`) so training reads pre-tokenized bytes instead of re-tokenizing.

---

## 5. Training plan

### Pipeline
1. `prepare_data.py` — download + curate + tokenize the corpus into token
   files.
2. `train.py` — nanoGPT-style training loop, fp16 AMP, periodic checkpointing,
   signal-handler suspend/resume.
3. `sample.py` — text generation from a checkpoint.
4. `benchmark.py` — measure real tokens/sec on this GPU; feed the real number
   back into the time estimates (do this *before* committing to a run length).

### Throughput expectation
Estimate: ~20–40k tokens/sec at 50M params on this 80 W-capped card.
- 1 epoch over 1 B tokens ≈ **~9–14 h**.
- Chinchilla-optimal 1-epoch run ≈ **~1–2 days**.
- Multi-epoch over-training ≈ **~3–5 days**.
**Benchmark first** — the 80 W cap / WSL2 driver can make real numbers 2–3×
worse than FLOP estimates.

### Run schedule
1. **Validate pipeline** on a small slice (~10–50 M tokens, ~1 epoch, overnight)
   to confirm it trains and text is coherent.
2. **Scale to full 1 B-token slice** and run the multi-day training with
   checkpointing + suspend/resume.

---

## 6. Suspend / resume & checkpointing

This is the key operational requirement: pause a multi-day run to free the GPU
for other work, then resume exactly where it left off.

### Semantics
- **Pause = checkpoint then exit.** Do NOT use `SIGSTOP` — that holds VRAM and
  defeats the purpose of freeing the GPU.
- **Resume = reload checkpoint and continue.**

### Checkpoint contents (`torch.save` of a dict)
- `model.state_dict()` — weights
- `optimizer.state_dict()` — Adam m/v (required; else momentum restarts)
- `step`, `epoch`, `tokens_seen` — resume position
- scheduler state, RNG states (`torch.get_rng_state()` + CUDA RNG)
- best_val / config / timestamp
- corpus slice id + tokenizer used (so resume matches the right data)

### Mechanics
- **Atomic save**: write to `ckpt.tmp` then `os.replace()` to final name, so a
  crash mid-write never corrupts the last good checkpoint.
- **Rotation**: keep `best.pt` + `last.pt`; also a `step-<N>.pt` every N steps.
  Delete old ones to bound disk.
- **Cadence**: checkpoint every ~1–2 h (or every N steps) automatically, plus on
  graceful signal.

### Signal handling
- `train.py` installs a `SIGINT`/`SIGTERM` handler: save checkpoint, then
  `sys.exit(0)`.
- **Ctrl-C to pause** → frees GPU. **Re-run with `--resume ckpt.pt`** to
  continue from the exact step, including optimizer and RNG state.

### Resume correctness
- `torch.load(..., weights_only=True, map_location=device)`.
- Reload model + optimizer state, set LR scheduler to the saved step, advance
  the data loader past `tokens_seen`.

---

## 7. Operational conventions (also in AGENTS.md)

- Python 3.14; use a project venv (`.venv/`) with `torch 2.14.0`.
- Commits: **assistant commits; user pushes** (per personal working agreement —
  `push` is the user's multi-repo script).
- Never commit large artifacts: `.venv/`, downloaded corpora, tokenized `.bin`
  files, checkpoints — all gitignored. Commit code + docs only.
- Corpus/checkpoints are large local-only files; ignore via `.gitignore` (this
  is a training project where the artifacts are the point, unlike personal
  reference copies).

---

## 8. Deliverables / milestones

1. Repo scaffold + docs (this file + AGENTS.md). ✅ (initial commit)
2. `prepare_data.py` + corpus downloader/curator.
3. `benchmark.py` → real tokens/sec.
4. `train.py` with checkpointing + suspend/resume.
5. `sample.py`.
6. Overnight validation run on a small slice.
7. Full 1 B-token run.