# toymodel

Personal toy language-model project: a ~50M-param decoder-only GPT trained on a
curated English, post-1900, STEM corpus (arXiv / Feynman / Wikipedia STEM /
STEM Q&A) using the local RTX 2080 Super Max-Q (8 GB) in WSL2. Authoritative
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
- GPU: RTX 2080 Super Max-Q, 8 GB VRAM, observed 80 W cap. nvidia-smi
  reports CUDA 12.7 driver capability. **Verified** working CUDA via torch;
  measured ~27k tokens/sec at 51M params, 8×256 micro-batch, 2.3 GB peak VRAM
  (single-step, no accumulation). Record sustained benchmark results once
  `benchmark.py` measures them.
- Model recipe: 8 layers × 512 dim, GPT-2 tokenizer (50,257 vocab), ~51 M
  params. Corpus shares (50/20/20/10) are data-loader sampling weights, not
  raw sizes. Textbook supply must be measured; Feynman is optional pending
  permission review. A 1 B-token budget does not guarantee convergence.

## Current state

- Repo scaffolded, docs written (`TOY_MODEL_PLAN.md`, this file).
- `.venv/` created; `torch 2.14.0+cu126` verified working on the GPU
  (`gpu_smoke.py` runs real fp16 forward/backward/AdamW at ~27k tok/s).
- **Not yet implemented**: `prepare_data.py`, `benchmark.py`, `train.py`,
  `sample.py`. No corpus downloaded, no training run started.
- See `TOY_MODEL_PLAN.md` §8 (milestones) for the next steps.

## Files

- `TOY_MODEL_PLAN.md` — full design: model, corpus, training plan,
  suspend/resume & checkpointing, operational conventions.
- `AGENTS.md` — this file.
- `gpu_smoke.py` — small 51M-param GPT smoke test; verifies the CUDA toolchain
  with a real fp16 training step (loss decreases, gradients finite).

## Planned files (from plan)

- `prepare_data.py` — download + curate + tokenize the corpus into token files.
- `benchmark.py` — measure real tokens/sec on this GPU.
- `train.py` — nanoGPT-style training loop, fp16 AMP, periodic checkpointing,
  signal-handler suspend/resume (`Ctrl-C` = checkpoint + exit; `--resume` =
  continue).
- `sample.py` — text generation from a checkpoint.
- `gpu_smoke.py` — implemented (see Files above).

## Suspend / resume (planned, not implemented)

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
