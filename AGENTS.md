# toymodel

Personal toy language-model project: a ~50M-param decoder-only GPT trained on a
curated English, post-1900, STEM corpus (arXiv / open textbooks / Wikipedia
STEM / STEM Q&A) using the local RTX 2080 Super Max-Q (8 GB) in WSL2. Authoritative
design is in `TOY_MODEL_PLAN.md`; this file records current state and working
rules. Keep changes small.

## Conventions

- **The assistant commits; the user pushes.** After the initial setup commit
  (already done), leave `git push` to the user. `push` is the user's shell
  script that pushes all their repos; don't substitute a plain `git push` when
  the user types a `(cd .. ; push )` line — run the script exactly as given.
- Python is **3.14.4**; use the project venv at `.venv/`. **Verified toolchain**:
  `torch==2.14.0+cu126` installed from the `download.pytorch.org/whl/cu126`
  index (NOT the PyPI default, which is `+cu130` and requires a newer driver
  than this machine's 12.7-capable driver). CUDA 12.6 runtime, cuDNN 9.10,
  numpy 2.5.3. Confirmed working sm_75 (Turing) fp16 training via
  `gpu_smoke.py`.
- **Never commit large artifacts**: `.venv/`, downloaded corpora, tokenized
  `.bin`/`.npy` files, checkpoints, logs. Commit code + docs only (see
  `.gitignore`). Corpus/checkpoints are the project's point, so they're
  gitignored rather than excluded.
- GPU: RTX 2080 Super Max-Q, 8 GB VRAM (≈1 GB held by Xwayland), observed
  80 W cap, compute capability 7.5. **Verified** working CUDA via torch;
  `gpu_smoke.py` measures ~28k tokens/sec at 51M params, 8×256 micro-batch,
  2.3 GB peak VRAM (20-step burst, no accumulation, synthetic data). Treat as
  an upper bound until `benchmark.py` measures a sustained run.
- Precision is **fp16 AMP**, not bf16: Turing has no bf16 tensor cores;
  measured bf16 autocast ≈3.6× slower (`is_bf16_supported()` is misleading).
- With tied embeddings, use nanoGPT-style init (std 0.02, scaled residual
  projections); PyTorch's default embedding init gives an initial loss of 50+.
- Model recipe: 8 layers × 512 dim, GPT-2 tokenizer (50,257 vocab), ~51 M
  params. Corpus shares (60/28/12) are data-loader sampling weights, not
  raw sizes. A 1 B-token budget does not guarantee convergence.
- **`train.py` is written and smoke-verified**: document Rajeev-style
  (nanoGPT-style) fp16 AMP loop, AdamW with weight-decay grouping (exclude
  dim<2), text-long cosine LR on `tokens_seen`, document-level mixture sampler
  (60/28/12 with SE split by tokens), local Generator RNG saved in every
  checkpoint for bit-exact resume, periodic + SIGINT/SIGTERM checkpointing
  with atomic save/fsync and rotation, held-out per-source eval. Verified:
  fresh run (loss 10.4 → 9.9 over ~1M tokens), `--resume last.pt`,
  SIGTERM-pause → clean save → resume, `--no-accumulate`. First-eval `best.pt`
  save needs `CKPT_DIR` to exist upfront (mkdir at startup). Checkpoint ≈
  0.61 GB (`last.pt` full state, `best.pt` ~0.2 GB weights-only).

## Current state

- Repo scaffolded, docs written (`TOY_MODEL_PLAN.md`, this file).
- `.venv/` created; `torch 2.14.0+cu126` verified working on the GPU
  (`gpu_smoke.py` runs real fp16 forward/backward/AdamW at ~28k tok/s).
- Corpus pipeline (`prepare_data.py`): **SE (~108M tokens, 140k threads)** and
  **arXiv ranked-direct (~152M GPT-2 tokens, 4,900 top-cited papers)** and
  **Wikipedia STEM-titled (~25M tokens, 19,596 articles, hf parquet streaming
  filtered on title)** all ingested end-to-end. Total supply ≈ 285 M tokens.
  Raw archives are deleted after cleaning (stream-and-discard; corpus/ keeps
  only clean + tokens; `audit` derives counts from corpus/tokens directly).
  Textbooks (OpenStax/Wikibooks) were **dropped** — not worth the engineering
  for the token gain; final **60/28/12 mix** (arXiv/SE/Wikipedia) stays within
  every 5-pass cap for a 1 B-token budget (see plan §3).
  - SE gotcha: dump filename uses site slugs — `math.stackexchange.com.7z`
    (not `mathematics`), `stats.stackexchange.com.7z` (not `statistics`).
  - lxml `iterparse` gotcha: must `dict(elem.attrib)` before `elem.clear()`;
    the live attrib dict gets emptied otherwise (kept 0 threads).
  - The HF RedPajama-1T repo is pointer-only (data at `data.together.xyz`,
    86 GB, raw LaTeX, no category) and `arxiv-papers-by-subject` is
    abstracts-only; ranked-direct is the cheap full-text route.
  - OpenAlex arXiv source id is `S4306400194`; `page` is capped at 50 so large
    ranking must use cursor pagination. The arXiv API returns versioned ids
    (`1412.6980v9`) — strip `v<N>` to match OpenAlex/extracted ids.
  - Citation-ranked arXiv skews to cs/cond-mat/quant-ph; math is ~2% of the
    list. The fetch alternates categories round-robin to keep math/stat.
  - `wiki-fetch` streams `wikimedia/wikipedia` `20231101.en`, keeps only
    STEM-titled articles (title regex, Wikipedia/List/disambiguation excluded),
    appends incrementally and stops at quota; unauthenticated HF can rate-limit
    (resume-safe — already-kept ids are skipped). Ran a couple of times to
    finish.
- **Not yet implemented**: `sample.py`. `benchmark.py` exists but the full
  sustained timed run is still to be done. The corpus is complete
  (arXiv/SE/Wikipedia, ~285M tokens). `train.py` is written and smoke-verified
  (see conventions above); next milestone is the sustained pilot run and
  `sample.py`.
- See `TOY_MODEL_PLAN.md` §8 (milestones) for the next steps.

## Files

- `TOY_MODEL_PLAN.md` — full design: model, corpus, training plan,
  suspend/resume & checkpointing, operational conventions.
- `AGENTS.md` — this file.
- `gpu_smoke.py` — small 51M-param GPT smoke test; verifies the CUDA toolchain
  with a real fp16 training step (loss decreases, gradients finite).
- `benchmark.py` — sustained tok/s / VRAM / power measurement (synthetic data,
  fp16 AMP, gradient accumulation, GradScaler). Sanity-checked at 15 s/micro-
  batch: ~31k tok/s, 2.5 GB (b8) / 4.1 GB (b16) peak alloc, ~74-77 W.
- `prepare_data.py` — download/extract/clean/tokenize/audit CLI. `se` and
  `arxiv-rank`/`arxiv-fetch` and `wiki-fetch` sources implemented end-to-end.
- `train.py` — nanoGPT-style training loop, fp16 AMP, periodic checkpointing,
  signal-handler suspend/resume (`Ctrl-C` = checkpoint + exit; `--resume` =
  continue).
- `sample.py` — text generation from a checkpoint (best.pt weights-only or
  last.pt full), temperature/top-k/top-p; rough-in pending recheck after the
  pilot run.

## Planned files (from plan)

(none — sample.py rough-in done; plan milestone 6: overnight validation run)

## Suspend / resume (implemented in `train.py`)

Pause a multi-day run = checkpoint then exit (never `SIGSTOP`, which holds
VRAM). Resume = `--resume ckpt.pt`. Checkpoints save model + optimizer (Adam
m/v), token-budget schedule, GradScaler, sampler and RNG state. Signal
handlers only set flags; save at a completed update boundary. See the plan
for atomic writes, compatibility checks and resume verification.

## Handoff procedure

End of a session. Triggered by the user saying "handoff". The agent does
everything here; the user pushes afterward — **never push**.

1. Check training processes and identify this project's run. Never blindly
   kill training. If a run is active, ask whether to leave it running or
   request graceful checkpoint-and-exit; verify the saved checkpoint before
   calling it paused.
2. Triage stray files (`git status`): add real content, gitignore recurring
   artifacts (corpus, `.bin`, checkpoints). Preserve unrelated work and
   checkpoints; do not delete ambiguous files to obtain a clean worktree.
3. Verify clean source: `git status` / `git diff`.
4. Commit source only (`git commit`, short imperative message, e.g. "Add
   corpus downloader").
5. Leave breadcrumbs in `TOY_MODEL_PLAN.md` or `AGENTS.md` BEFORE committing:
   non-obvious findings (real benchmark numbers, dataset gotchas) go in the
   docs. `AGENTS.md` gets only facts that change how a future session works.

## Sibling repos

- `~/raylib-test` — personal raylib playground; documents WSLg/D3D12 GPU access.
- `~/system-administration` — private; host-level notes, agent guidance
  (source of the working agreements above), and the `push` script at
  `bin/push`.
- `~/collatz`, `~/p_vs_np`, `~/factor-circuit` — other public personal projects.

`~/.local/bin/push` (symlinked to `system-administration/bin/push`) pushes every
repo whose origin is `github.com/david-battle/*`. Do not push unless asked.
