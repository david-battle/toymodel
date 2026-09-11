# toymodel

Personal toy language-model project: a ~50M-param decoder-only GPT trained on a
curated English, post-1900, STEM corpus (arXiv / open textbooks / Wikipedia
STEM / STEM Q&A) using the local RTX 2080 Super Max-Q (8 GB) in WSL2. Authoritative
design is in `TOY_MODEL_PLAN.md`; this file records current state and working
rules. Keep changes small.

## Conventions

- **NEVER `git push`.** The assistant commits; the user pushes. This is absolute — do not push unless explicitly asked. `push` is the user's shell script that pushes all their repos; when the user types `(cd .. ; push )`, run that exact script, don't substitute `git push`.
- **The assistant commits; the user pushes.** After the initial setup commit (already done), leave `git push` to the user. `push` is the user's shell script that pushes all their repos; don't substitute a plain `git push` when the user types a `(cd .. ; push )` line — run the script exactly as given.
- **Start background training only via `./run_training.sh` or `./run_training_watchdog.sh`.** A bare
  `python train.py &` (or `nohup ... &`) keeps the launching shell/tool open
  waiting on the child's inherited stdout and trips timeouts. The scripts use
  `setsid` + a log redirect of ALL three fds (`>log 2>&1 </dev/null`), which
  detaches the run into its own session so it survives the shell that started
  it.
  - `./run_training.sh`: basic launcher (defaults: pilot `--eval-steps 75 --ckpt-steps 100`; train.py defaults 200/500). Pause with `kill -TERM $(cat logs/pilot.pid)` (checkpoint + exit); resume with `--resume ckpt/last.pt`.
  - `./run_training_watchdog.sh`: launches training + a watchdog that monitors throughput via `logs/pilot.log`. If tok/s drops below threshold (default 10k) for 3 consecutive checks, sends SIGTERM to trigger checkpoint, waits, then restarts with `--resume ckpt/last.pt`. Watchdog logs to `logs/watchdog.log`, PID in `logs/watchdog.pid`. Stop watchdog with `kill -TERM $(cat logs/watchdog.pid)`.
- **Stop watchdog BEFORE stopping training when using `run_training_watchdog.sh`.** The watchdog monitors `logs/pilot.log` and restarts training if it detects the process died or throughput drops. If you stop training first, the watchdog will see the process gone and restart it on the next log line (or immediately if it polls). Always: `kill -TERM $(cat logs/watchdog.pid)` then `kill -TERM $(cat logs/pilot.pid)`.
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
- Corpus pipeline (`prepare_data.py`): **SE (~48M tokens, 76k threads)** and
  **arXiv ranked-direct (~54M GPT-2 tokens, 5,015 top-cited papers)** and
  **Wikipedia STEM-titled (~24.6M tokens, 19,596 articles, hf parquet streaming
  filtered on title)** all ingested end-to-end. Total supply ≈ 127 M tokens
  (LaTeX-stripped). Raw archives are deleted after cleaning (stream-and-discard;
  corpus/ keeps only clean + tokens; `audit` derives counts from corpus/tokens
  directly). Textbooks (OpenStax/Wikibooks) were **dropped** — not worth the
  engineering for the token gain; final **60/28/12 mix** (arXiv/SE/Wikipedia)
  exceeds 5-pass cap at 2.5B budget (arXiv ~28×, SE ~15×, Wiki ~12×); 1B budget respects caps.
  - LaTeX stripping (pylatexenc + custom heuristics): converts `$...$`,
    `$$...$$`, `\frac`, `\sqrt`, Greek letters, arrows, relations → Unicode.
    Strips `## Answer` markers from SE.
  - **Audit note**: 285M→127M token drop from LaTeX stripping is plausible but doc-count diff vs pre-strip not yet verified; could include silently dropped documents.
- **50M-token pilot complete** (50.1M tokens, best_val 5.68, loss 10.4→5.5,
  0 AMP skips, finished cleanly on `budget-exhausted`).
- **124M full run CRASHED at ~1017M tokens (step 7760)** — CUDA unknown error (likely OOM / driver fault at 86°C, 7.8 GB VRAM).
  Last clean checkpoint: **step 7500, 983M tokens** (`ckpt/last.pt`, 1.49 GB). ~34M tokens lost.
  **Recovery**: 983M checkpoint was overwritten by accidental fresh run. Recovered model weights from `best.pt` (best_val=2.899 at 983M) into a fresh `last.pt` with reset optimizer/scaler/RNG at step 0. Training restarts from 0 tokens with warm-started weights.
  **Watchdog implemented** (`watchdog.py`, `run_training_watchdog.sh`): monitors `logs/pilot.log` for throughput drops below 10k tok/s (3 consecutive checks), sends SIGTERM for graceful checkpoint+restart.
  **Checkpoint rotation improved**: step snapshots now every 5k steps (was 10k), keeping 5 most recent.
  Target: 2.5B tokens (19.7 epochs).
- **Run paused at step 4621 (605.7M tokens, 24.2%)** — graceful SIGTERM checkpoint saved to `ckpt/last.pt` (1.49 GB). Latest eval (step 4500, 589.8M): agg 2.564 (arxiv 2.251, se-math 2.331, wiki 3.255). Watchdog stopped. Resume with `--resume ckpt/last.pt`.
- **Fixed resume config bug**: `run_training_watchdog.sh` now appends CLI args to defaults (not replace), so `--resume` keeps full 124M config. `watchdog.py` reads training cmdline from `/proc/<pid>/cmdline` for restarts.
- `train.py` updated: model config (n_layer, n_head, n_embd, block) now
  configurable via CLI; `--resume` restores full state bit-exactly.
- `prepare_data.py` updated: `latex-strip` and `retokenize` subcommands added.
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
- `watchdog.py` — monitors `logs/pilot.log` throughput; on sustained drop
  below threshold, sends SIGTERM to training process for graceful checkpoint+restart.
- `run_training_watchdog.sh` — launches training + watchdog detached via `setsid`.

## Planned files (from plan)

(none — sample.py rough-in done; plan milestone 6: overnight validation run)

## Suspend / resume (implemented in `train.py`)

Pause a multi-day run = checkpoint then exit (decided: **no** `SIGSTOP`, **no**
resident-in-VRAM pause — a paused run always checkpoints and exits, freeing
VRAM and terminating the process). Resume = `--resume ckpt.pt`. Checkpoints
save model + optimizer (Adam m/v), token-budget schedule, GradScaler, sampler
and RNG state. Signal handlers only set flags; save at a completed update
boundary. See the plan for atomic writes, compatibility checks and resume
verification.

**Watchdog integration**: `watchdog.py` uses the same SIGTERM mechanism. It
tails `logs/pilot.log`, parses `tok/s=` from log lines, and when throughput
drops below threshold (default 10k) for 3 consecutive readings (30s interval),
sends SIGTERM. Training checkpoints at next update and exits. Watchdog waits,
then restarts with `--resume ckpt/last.pt`. This recovers from throughput
degradation before OOM/crash.

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
