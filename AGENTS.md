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
- Python is **3.14.4**; use the project venv at `.venv/` with `torch 2.14.0`
  (the only torch release supporting 3.14). Never assume a package is present —
  check before using.
- **Never commit large artifacts**: `.venv/`, downloaded corpora, tokenized
  `.bin`/`.npy` files, checkpoints, logs. Commit code + docs only (see
  `.gitignore`). Corpus/checkpoints are the project's point, so they're
  gitignored rather than excluded.
- GPU: RTX 2080 Super Max-Q, 8 GB VRAM, CUDA 12.7, reachable from WSL2 via
  torch CUDA. Power-capped at 80 W — expect ~20–40k tokens/sec at 50M params
  (benchmark before committing to a run length).

## Current state

- Repo scaffolded, docs written (`TOY_MODEL_PLAN.md`, this file).
- **Not yet implemented**: `prepare_data.py`, `benchmark.py`, `train.py`,
  `sample.py`. No venv yet, no corpus downloaded, no training run started.
- See `TOY_MODEL_PLAN.md` §8 (milestones) for the next steps.

## Files

- `TOY_MODEL_PLAN.md` — full design: model, corpus, training plan,
  suspend/resume & checkpointing, operational conventions.
- `AGENTS.md` — this file.

## Planned files (from plan)

- `prepare_data.py` — download + curate + tokenize the corpus into token files.
- `benchmark.py` — measure real tokens/sec on this GPU.
- `train.py` — nanoGPT-style training loop, fp16 AMP, periodic checkpointing,
  signal-handler suspend/resume (`Ctrl-C` = checkpoint + exit; `--resume` =
  continue).
- `sample.py` — text generation from a checkpoint.

## Suspend / resume (as implemented in the plan)

Pause a multi-day run = checkpoint then exit (never `SIGSTOP`, which holds
VRAM). Resume = `--resume ckpt.pt`. Checkpoints save model + optimizer (Adam
m/v) + step/epoch/tokens_seen + scheduler + RNG state; written atomically
(tmp + `os.replace`) with `best.pt` / `last.pt` / step rotation.

## Handoff procedure

End of a session. Triggered by the user saying "handoff". The agent does
everything here; the user pushes afterward — **never push**.

1. Kill stray training processes if any (check `pgrep` for `train.py`).
2. Triage stray files (`git status`): add real content, gitignore recurring
   artifacts (corpus, `.bin`, checkpoints), delete junk. Never ask.
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