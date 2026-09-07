# Toy Language Model — Full Plan

This document is the authoritative design for a small, GPU-trained English
language model focused on post-1900 STEM text. It covers the model
architecture, the curated corpus, the training plan (including suspend/resume),
and operational conventions. Companion file: `AGENTS.md` (current state +
working rules).

---

## 1. Hardware & environment (initial observations)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Super (Max-Q), **8 GB VRAM** |
| Driver / CUDA | Driver 566.03; nvidia-smi reports CUDA 12.7 driver capability |
| Host | WSL2 (Linux 6.18 msft), ~27 GB RAM |
| Python | 3.14.4 |
| torch | 2.14.0 proposed; package index listing seen, installation untested |
| Disk | ~936 GB free |

**GPU access note**: nvidia-smi sees the GPU in WSL2; this does not yet verify
PyTorch CUDA execution. Its CUDA version is not an installed toolkit version.
In a project venv, verify wheel support for Python 3.14 and the GPU's Turing
architecture (sm_75), driver/runtime compatibility, and a real fp16
forward/backward/AdamW step. Record working versions; do not upgrade host
drivers as part of setup without approval. The observed power cap is 80 W,
not a WSL2-specific limit. Display applications also consume VRAM.

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
| Micro-batch | Start at 4-8 sequences x 256 tokens; benchmark up to 32 |
| Effective batch | ~131k tokens/update via gradient accumulation |
| LR | cosine schedule, peak ~6e-4, min ~6e-5, warmup 2% of run tokens |

Use pre-LayerNorm blocks, GELU MLPs of width 4x the embedding dimension,
learned positional embeddings, causal attention, tied input/output weights,
and initially zero dropout. Exclude biases and LayerNorm weights from weight
decay. These choices make the parameter estimate reproducible.

**Parameter count check**: non-embedding params ≈ 12·n_layer·n_embd² =
12·8·512² ≈ 25 M; embedding (tied with the LM head) ≈ 50,304·512 ≈ 26 M.
Total ≈ 51 M. (At n_embd=768 the same recipe is ≈57 M + 39 M ≈ 95 M — nearly
double the target.) Half the parameters live in the embedding table, which is
normal at this scale; a smaller vocab would shift capacity into the blocks but
GPT-2's tokenizer is the path of least resistance.

**Memory model (AMP + AdamW ≈ 16 bytes/param)**: fp32 master weights (4) +
fp32 grads (4) + Adam m/v (8) → 51 M × 16 ≈ 0.8 GB, before AMP casts,
activations, logits, loss temporaries, allocator and CUDA workspaces. At an
8k-token micro-batch, logits alone contain ~412 M elements (~0.82 GB fp16 or
1.65 GB fp32). Thus 32 sequences is a benchmark candidate, not a guaranteed
fit. Start with 4-8 sequences and increase while measuring peak allocated and
reserved VRAM, leaving headroom for the display. Use accumulation to reach the
effective batch without retaining graphs between micro-steps.

**Batch and schedule**: ~131k tokens/update is a starting experiment, not a
requirement; smaller batches are not inherently wrong. Divide each
micro-batch loss by accumulation count, unscale gradients, then clip global
norm to 1.0 before the optimizer update. Track AMP-skipped updates separately.
Define warmup in tokens: 2% of the selected run budget, followed by cosine
decay over the remaining budget. The earlier fixed 500-step warmup would
exceed a 10-50 M-token pilot at this batch size. Count consumed tokens even on
AMP-skipped updates and use that counter for the schedule; frequent skips or
nonfinite loss require diagnosis, not silent continuation.

**Why 256 context**: a cheap baseline for local coherence, not a claim that
longer context dilutes capacity. It truncates many STEM arguments and Q&A
pairs. Benchmark 512 as a possible alternative before fixing the full-run
architecture; changing learned positional embeddings is not an exact resume.

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

**Supply vs. demand.** All supply figures above are rough, unverified estimates
before license filtering, cleaning and deduplication, not acquisition promises.
For a 1 B-token training budget the source allocations are 500/200/200/100 M
token presentations, not necessarily distinct tokens. Measure unique eligible
tokens U per source and report expected exposure as allocation/U. Sample books
within the textbook slot in proportion to eligible token counts, not equally
by book; Feynman is optional pending permissions. Provisionally cap expected
exposure at 5 passes per source/book. If supply falls short, add eligible
expository sources or shorten the pilot; discuss changing the 50/20/20/10 mix
or raising the cap before the full run. Repetition is not equivalent to fresh
data, and high quality does not prevent memorization.

### Curation goals
The user explicitly wants the *best* papers/articles/questions in the slice,
not a blind dump. Curation strategy:

1. **arXiv**: first audit a sample of an existing full-text dataset (e.g. the
   arXiv subset of RedPajama) for extraction quality, stable paper IDs,
   version/date/category metadata and per-paper licenses. Do not assume these
   fields survive preprocessing; reject candidates lacking a reliable join to
   source metadata. If suitable, use it and
   filter by category (`physics.*`, `math.*`, `cs.*`, `quant-ph`, `cond-mat`,
   `astro-ph`). arXiv itself carries no citation counts; rank/filter by
   age- and field-normalized citation signals from OpenAlex or Semantic
   Scholar, subject to current access/rate limits. Citation count is not
   correctness: retain surveys, tutorials and a recent-paper slice within
   subject quotas rather than only top-cited papers. Drop bibliographies and
   author boilerplate; preserve readable equations and definitions. Reject
   badly extracted text, not mathematics merely because it contains formulas.
2. **Wikipedia**: from a current dump, keep articles under the STEM category
   trees (walk 2–3 levels down from Physics, Mathematics, Computer science,
   Chemistry, Biology). Rank by quality signals: Featured/Good article status,
   article length, and incoming link count. Category graphs are cyclic and
   leak into non-STEM topics: use a visited set and audit topic relevance.
   Length/popularity alone is not quality. Strip markup and reference lists,
   but preserve meaningful prose, captions and mathematical definitions.
3. **Textbooks**: eligible OpenStax and Wikibooks STEM works; Feynman only
   after permission review. Check extraction rather than assuming clean HTML.
4. **Q&A**: StackExchange data dumps (CC-BY-SA). Keep question + accepted (or
   top-voted) answer, with score >= 5 as a pilot heuristic, not proof of
   correctness. Reject obsolete or contradicted answers during spot checks;
   balance sites/topics so StackOverflow does not dominate. Verify current
   dump access and terms. Exclude Reddit to keep acquisition and provenance
   simple, not because all historical dumps are necessarily unavailable.

For all sources: require predominantly English prose and publication/revision
after 1900 (historical subject matter is allowed). Record source ID, URL,
date/version, license, retrieval date, content hash and filter decisions in a
manifest. Inspect stratified samples from each source and both accepted and
rejected documents before freezing a corpus. Prefer diversity and clear
explanations over popularity alone; remove obvious personal contact details.

### Licensing
arXiv papers carry per-paper licenses (many are arXiv-perpetual-nonexclusive,
not CC-BY); OpenStax and Wikibooks are CC-BY / CC-BY-SA; Wikipedia and
StackExchange are CC-BY-SA; the online Feynman Lectures are Caltech-copyright,
free to read but not thereby licensed for unrestricted reuse. Personal use
does not automatically settle permission to scrape or train. Check each
source's actual license/version and acquisition terms; default to explicitly
permitted material and omit uncertain sources pending review. A dataset-level
license does not override underlying article rights. Preserve required
attribution and do not redistribute the corpus. Review obligations separately
before any future public release of checkpoints or substantial text excerpts.

### Target size
- **Start (overnight validation)**: ~10–50 M tokens.
- **Full run**: ~1 B training-token presentations (about 20 x 50 M params).

The 20:1 heuristic comes from compute-optimal scaling experiments, not a
convergence theorem, a required number of epochs, or a guarantee for a tiny
STEM model. A weighted, repeated mixture is not one pass over 1 B unique
tokens. Use held-out loss and sample quality to decide whether more training
is useful; expect plausible STEM-style completions, not reliable reasoning or
an instruction-following assistant.

Keep the raw slice separate from the tokenized training set so re-tokenizing
with a different vocab doesn't require re-downloading. Hold out ~0.5 % of each
source as validation and another ~0.5% as a final untouched test set, enlarged
where needed to cover topics. Deduplicate exact and near-duplicate content
across sources before splitting. Keep paper versions, mirrored articles,
whole books (or related chapter groups) and entire Q&A threads in one split.
Save fixed split manifests. A handful of random paragraphs from the same book
is not an independent evaluation set.

---

## 4. Tokenization

Use `tiktoken` `gpt2` (50,257-token BPE). A BPE vocabulary cannot simply be
"resized" — trimming merges changes the tokenization — so pick a tokenizer
that is already the right size rather than `cl100k_base` (~100k tokens, which
would double the embedding table). Tokenize the curated corpus offline to
`uint16` `.bin` files per source (like nanoGPT's `prepare.py`) so training
reads pre-tokenized bytes via `np.memmap` instead of re-tokenizing. Append an
end-of-text token between documents.
Pad the model table, not the tokenizer: mask logits for IDs 50,257-50,303
before loss and sampling so unused IDs cannot be generated. Encode source
text as ordinary text (including literal special-token strings), then append
the actual EOT ID. Store document offsets alongside token files for auditing.

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
Training cost is roughly 6*N FLOPs/token (~306 MFLOP at 51 M), but this
ignores attention overhead and cannot predict achieved throughput. The earlier
15-35k tokens/sec range assumes unmeasured useful compute and is not a planning
guarantee. Sensitivity for 1 B consumed tokens:

| Measured training rate | GPU training time |
|---|---|
| 2k tokens/sec | 5.8 days |
| 5k tokens/sec | 2.3 days |
| 10k tokens/sec | 27.8 hours |
| 25k tokens/sec | 11.1 hours |

Add preparation, validation, checkpoint time and pauses separately. Benchmark
the actual model, vocabulary projection, loss, backward, accumulation and
optimizer, not just forward passes. Warm up and synchronize CUDA timing;
measure sustained performance for 10-20 minutes with power, temperatures and
peak VRAM. A single 77 C observation does not establish thermal throttling.
Start eager; try SDPA, fused AdamW and torch.compile only if supported and
measurably useful. Do not assume Turing supports newer FlashAttention kernels.

### Run schedule
1. **Validate pipeline**: first overfit a tiny batch and test checkpoint/resume;
   then run a 10-50 M-token pilot with its own token-budget schedule. Check
   finite/decreasing loss, AMP stability, and held-out samples; coherent prose
   is not a guaranteed pilot outcome.
2. **Scale to full 1 B-token slice** and run the multi-day training with
   checkpointing + suspend/resume.

Evaluate fixed held-out batches per source about every 10 M tokens (more
often in the pilot), with eval mode and no gradients. Log per-source loss,
the fixed 50/20/20/10 weighted aggregate, train/validation gap, tokens/sec and
AMP skips. Choose best.pt using that fixed aggregate. Use isolated evaluation
and sampling RNGs so evaluation does not change training data order. Keep a
fixed prompt set for qualitative comparisons; consult the test set only at
the end. Preserve the initial full-run schedule when resuming; extending its
budget is a new experiment, not an invisible scheduler reset.

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
- per-source data-loader position/RNG state and mixture-sampler RNG, plus
  Python/NumPy (if used) and `torch`/CUDA RNG states
- best_val / config / timestamp
- corpus/split/tokenizer hashes, architecture, batch/accumulation, mixture,
  total schedule budget, checkpoint format version and code/package versions

Size: ~51 M params × (4 B weights + 8 B Adam) ≈ **0.6 GB per checkpoint**, so
rotation matters even with 900 GB free.

### Mechanics
- **Atomic save**: write a temporary file in the destination directory, flush
  and fsync it, then `os.replace()` the final name (fsync the directory where
  supported). Atomic replacement prevents partial-file visibility; durability
  against power loss also depends on storage. On save failure, report the
  error, preserve the last good file and do not report a successful pause.
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
- Load a trusted checkpoint with `weights_only=True, map_location="cpu"` to
  avoid duplicate GPU state and keep CPU RNG tensors on CPU. Encode NumPy
  state as tensors/plain values rather than raw NumPy objects; test restricted
  loading. Move model/optimizer tensors to their required devices explicitly.
- Reload model, optimizer, and GradScaler state; recompute the LR from the
  saved `tokens_seen`; restore each source's loader position.
- Initialize objects first, then restore RNG immediately before the next
  batch. Use a simple synchronous sampler initially, avoiding unsaved worker
  prefetch queues. Save only after accumulation, optimizer/scaler updates and
  gradient clearing have completed.
- Validate configuration and data hashes on load; reject mismatches by default.
  Batch/mix changes require an explicit new-run override. Unchanged config is
  necessary but not sufficient for bit-exact replay: kernels and software may
  be nondeterministic. Test uninterrupted versus interrupted runs in a
  deterministic small configuration, comparing next batch, LR, counters,
  scaler, optimizer and weights. Normal training promises stateful continuation,
  not unconditional bitwise identity.

---

## 7. Operational conventions (also in AGENTS.md)

- Python 3.14; use a project venv (`.venv/`) with `torch 2.14.0`.
- Commits: **assistant commits; user pushes** (per personal working agreement —
  `push` is the user's multi-repo script).
- Never commit large artifacts: `.venv/`, downloaded corpora, tokenized `.bin`
  files, checkpoints — all gitignored. Commit code + docs only.
- Training artifacts are project outputs, ignored via `.gitignore`.

---

## 8. Deliverables / milestones

1. Repo scaffold + docs (this file + AGENTS.md). ✅ (initial commit)
2. Minimal model + CUDA training smoke test, then `benchmark.py` for real tokens/sec.
3. `prepare_data.py` + corpus downloader/curator and measured supply audit.
4. `train.py` with checkpointing + suspend/resume.
5. `sample.py`.
6. Overnight validation run on a small slice.
7. Full 1 B-token run.
