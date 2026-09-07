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
| `n_embd` | 512 |
| `block_size` (context) | 256 |
| `vocab_size` | 50,257 (GPT-2 BPE via `tiktoken` `gpt2`), padded to 50,304 |
| Params (approx) | ~51 M (≈25 M transformer blocks + ≈26 M tied embedding/LM head) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1) |
| Precision | fp16 mixed precision (AMP + GradScaler; Turing has no bf16) |
| Micro-batch | 32 sequences × 256 tokens = 8,192 tokens |
| Effective batch | ~131k tokens/step via gradient accumulation (×16) |
| LR | cosine schedule, peak ~6e-4, min ~6e-5, warmup ~500 steps |

**Parameter count check**: non-embedding params ≈ 12·n_layer·n_embd² =
12·8·512² ≈ 25 M; embedding (tied with the LM head) ≈ 50,304·512 ≈ 26 M.
Total ≈ 51 M. (At n_embd=768 the same recipe is ≈57 M + 39 M ≈ 95 M — nearly
double the target.) Half the parameters live in the embedding table, which is
normal at this scale; a smaller vocab would shift capacity into the blocks but
GPT-2's tokenizer is the path of least resistance.

**Memory model (AMP + AdamW ≈ 16 bytes/param)**: fp32 master weights (4) +
fp32 grads (4) + Adam m/v (8) → 51 M × 16 ≈ 0.8 GB. Activations at an 8k-token
micro-batch add a few GB; well within 8 GB. If benchmarking shows headroom,
raise the micro-batch (fewer accumulation steps) rather than the effective
batch.

**Why ~131k tokens/step**: 4k-token steps would mean ~244k noisy optimizer
steps over 1 B tokens. ~128k tokens/step (≈8k steps for 1 B tokens) is in the
normal range for a model this size; the cosine schedule should be set by
`tokens_seen`, not step count, so the batch can change across resumes.

**Why 256 context**: small model, small corpus; longer context dilutes the
limited weight capacity. 256 is enough to learn local coherence and basic
grammar.

---

## 3. Corpus (the 50/20/20/10 mix)

Lean heavily English, post-1900, STEM. Shares are **sampling weights** for the
data loader (fraction of training tokens drawn from each source), not raw
corpus sizes — this lets a small high-quality source be upweighted without
physically duplicating it.

| Share | Source | Role | Raw supply |
|---|---|---|---|
| **50%** | **arXiv papers** (physics, math, CS) | STEM backbone | billions of tokens — ample |
| **20%** | **Expository textbooks**: Feynman Lectures + OpenStax STEM texts (CC-BY) + Wikibooks STEM | clean explanatory prose | Feynman ≈ 2 M tokens; OpenStax ≈ 10–20 M; Wikibooks ≈ 10 M+ |
| **20%** | **English Wikipedia STEM articles** (physics/math/CS/chem/bio categories) | broad modern vocabulary | hundreds of millions — ample |
| **10%** | **StackExchange STEM Q&A** (physics, math, cs, stats, chemistry, StackOverflow) | conversational Q&A style | hundreds of millions — ample |

**Supply vs. demand.** At the full 1 B-token run, 20 % = 200 M tokens. The
Feynman Lectures alone are only ~2 M tokens, so they cannot fill that slot (the
original plan effectively asked for 100 epochs of Feynman). The textbook slot
is therefore shared with other open expository sources, and the expository slot
as a whole will still be repeated a few times (~5–10 epochs) at 1 B tokens.
That is acceptable for a small high-quality source but should be watched: if
validation loss on held-out Feynman text starts rising while the others fall,
lower the weight. At the overnight scale (10–50 M tokens) supply is not a
problem for any source.

### Curation goals
The user explicitly wants the *best* papers/articles/questions in the slice,
not a blind dump. Curation strategy:

1. **arXiv**: do not start from raw LaTeX. Use an existing cleaned full-text
   arXiv dataset from Hugging Face (e.g. the arXiv subset of RedPajama) and
   filter by category (`physics.*`, `math.*`, `cs.*`, `quant-ph`, `cond-mat`,
   `astro-ph`). arXiv itself carries no citation counts; rank/filter by
   citation count via the OpenAlex or Semantic Scholar API (both free) and
   keep the top-N per category. Drop reference sections, author blocks, and
   papers whose text is mostly equations or tables.
2. **Wikipedia**: from a current dump, keep articles under the STEM category
   trees (walk 2–3 levels down from Physics, Mathematics, Computer science,
   Chemistry, Biology). Rank by quality signals: Featured/Good article status,
   article length, and incoming link count. Strip templates/infoboxes/refs.
3. **Textbooks**: Feynman (minimal cleaning; strip figure captions/equation
   numbers), OpenStax (CC-BY, clean HTML/XML), Wikibooks STEM shelves.
4. **Q&A**: StackExchange data dumps (CC-BY-SA). Keep question + accepted (or
   top-voted) answer, score ≥ 5, dedupe, drop code-only posts. Reddit is
   dropped: Pushshift bulk dumps are no longer available and the API is
   restricted.

### Licensing
arXiv papers carry per-paper licenses (many are arXiv-perpetual-nonexclusive,
not CC-BY); OpenStax and Wikibooks are CC-BY / CC-BY-SA; Wikipedia and
StackExchange are CC-BY-SA; the online Feynman Lectures are Caltech-copyright,
free to read but not redistributable. This is fine for a personal
training experiment; keep sources in separate directories so provenance is
clear, and do not redistribute the corpus.

### Target size
- **Start (overnight validation)**: ~10–50 M tokens.
- **Full run**: ~1 B tokens (20 × 50 M params, Chinchilla-optimal).

Keep the raw slice separate from the tokenized training set so re-tokenizing
with a different vocab doesn't require re-downloading. Hold out ~0.5 % of each
source (by document, not by token) as validation.

---

## 4. Tokenization

Use `tiktoken` `gpt2` (50,257-token BPE). A BPE vocabulary cannot simply be
"resized" — trimming merges changes the tokenization — so pick a tokenizer
that is already the right size rather than `cl100k_base` (~100k tokens, which
would double the embedding table). Tokenize the curated corpus offline to
`uint16` `.bin` files per source (like nanoGPT's `prepare.py`) so training
reads pre-tokenized bytes via `np.memmap` instead of re-tokenizing. Append an
end-of-text token between documents.

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
Training cost ≈ 6·N FLOPs/token ≈ 300 MFLOP/token at N = 51 M. An 80 W
Max-Q Turing card sustaining ~5–10 TFLOPS of useful fp16 tensor-core work gives
**~15–35k tokens/sec**; this is an estimate, not a measurement.
- 1 B tokens at 25k tok/s ≈ **~11 h**; at 15k tok/s ≈ **~19 h**.
- Chinchilla-optimal single pass over 1 B tokens ≈ **~0.5–1 day** of GPU time,
  realistically **1–2 days** wall-clock with pauses.
- Multi-epoch over-training (3–5 passes) ≈ **~3–5 days**.
**Benchmark first** — the 80 W cap, thermal throttling (the card was already
at 77 °C idle-ish), and WSL2 overhead can make real numbers 2–3× worse than
FLOP estimates. Use `torch.compile` and fused AdamW if they work on this
torch/CUDA combination; check `nvidia-smi` clocks during the benchmark.

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
- `step`, `tokens_seen` — resume position (the LR schedule keys off
  `tokens_seen`)
- `GradScaler` state (fp16 AMP loss scale)
- per-source data-loader position/RNG state, plus `torch`/CUDA RNG states
- best_val / config / timestamp
- corpus slice id + tokenizer used (so resume matches the right data)

Size: ~51 M params × (4 B weights + 8 B Adam) ≈ **0.6 GB per checkpoint**, so
rotation matters even with 900 GB free.

### Mechanics
- **Atomic save**: write to `ckpt.tmp` then `os.replace()` to final name, so a
  crash mid-write never corrupts the last good checkpoint.
- **Rotation**: keep `best.pt` + `last.pt`; also a `step-<N>.pt` every N steps,
  keeping only the most recent few.
- **Cadence**: checkpoint every ~30–60 min (or every N steps) automatically,
  plus on graceful signal. A save takes a few seconds.

### Signal handling
- `train.py` installs a `SIGINT`/`SIGTERM` handler that only sets a flag; the
  training loop checks the flag at the end of each optimizer step, saves a
  checkpoint, and exits. (Saving from inside the handler mid-step could
  capture a half-updated optimizer.) A second Ctrl-C forces immediate exit
  without saving.
- **Ctrl-C to pause** → frees GPU. **Re-run with `--resume ckpt.pt`** to
  continue from the exact step, including optimizer and RNG state.

### Resume correctness
- `torch.load(..., weights_only=True, map_location=device)` (the checkpoint
  holds only tensors and plain Python values, so this works).
- Reload model, optimizer, and GradScaler state; recompute the LR from the
  saved `tokens_seen`; restore each source's loader position.
- Resuming is only bit-exact if the batch size and data mix are unchanged;
  changing them is allowed but should be logged.

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