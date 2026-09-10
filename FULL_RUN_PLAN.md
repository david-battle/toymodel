# Full Run Plan: 124M GPT on 2.5B Tokens

**Target**: 124M-param decoder-only GPT (L12 H12 D768), context 512, trained on 2.5B tokens of cleaned STEM corpus with instruction-tuning phase.

**Hardware**: RTX 2080 Super Max-Q 8GB (≈7 GB usable after display; measured peak 7.8 GB)

---

## Model Configuration

| Param | Value |
|-------|-------|
| n_layer | 12 |
| n_head | 12 |
| n_embd | 768 |
| block_size | 512 |
| vocab_size | 50,304 (GPT-2 + padding) |
| Params | ~124M (non-embed: ~67M, embed: ~57M tied) |
| Precision | fp16 AMP (Turing tensor cores) |

---

## Training Config

| Setting | Value |
|---------|-------|
| Micro-batch | 4 × 512 = 2,048 tokens |
| Gradient accum | 64 → 131,072 tokens/update |
| LR schedule | Cosine: 6e-4 → 6e-5 over 2.5B tokens |
| Warmup | 2% (50M tokens) |
| Weight decay | 0.1 (excl. bias/LN/embed) |
| Grad clip | 1.0 |
| Optimizer | AdamW (β=0.9, 0.95) |
| Eval interval | ~65M tokens (every 500 steps) |
| Checkpoint interval | ~65M tokens (every 500 steps) + signals |

**VRAM**: ~7.8 GB peak allocated (measured), near-zero headroom on 8 GB Max-Q; GPU temp 86°C sustained.

**Throughput**: ~11–13.6k tok/s sustained (measured, after fresh restart). **Degradation to 2–11k tok/s observed due to CUDA context fragmentation; requires manual process restart to recover. No automated watchdog implemented.**

**Time estimate**: 2.5B / 11k ≈ **63 hours (2.6 days)** pretrain + 4h instruct-tune = **~3 days GPU time**. 2-week wall budget allows for pauses, data prep, eval.

---

## Data Pipeline

### Phase 1: LaTeX Stripping & Re-tokenization ✅ DONE (2026-09-09)
- Parsed `corpus/clean/*.jsonl` with `pylatexenc` + custom heuristics
- Converted LaTeX math → Unicode/natural language
- Stripped `## Answer` markers from SE data
- Re-tokenized with tiktoken GPT-2 → `corpus/tokens/*.bin`
- **Result**: 127M tokens (arXiv 54M, SE 48M, Wikipedia 24.6M)

### Phase 2: Pretraining 🔄 IN PROGRESS (started 2026-09-09)
- 2.5B token budget (19.7 epochs over 127M unique tokens)
- Mixture: arXiv 60% / SE 28% / Wikipedia 12%
- Held-out: 0.5% per source (fixed split)
- Resume from `ckpt/last.pt` on interrupt
- **Current (2026-09-10 11:47)**: 891M tokens (35.7%), step 6800, loss ~2.9, best_val 2.91 at 852M (step 6500)
- **Throughput**: stable 11–14k tok/s
- **Next eval**: 918M tokens (step 7000)
- **Next checkpoint**: 918M tokens (step 7000)

### Phase 3: Instruction Tuning (planned)
- Format SE Q&A as chat: `Q: <question>\nA: <accepted_answer><|endoftext|>`
- Filter: score ≥ 5, accepted answer exists
- ~100k examples × 3 epochs = ~300M tokens
- LR: 1e-4 constant (no decay), µ-batch 8, accum 32
- Start from best pretrain checkpoint

### Phase 4: Evaluation (planned)
- Perplexity on held-out test sets (per source + aggregate)
- Generation quality: fixed prompts, temperature sweep
- Coherence metrics: repetition rate, EOT emission, LaTeX artifacts
- Comparison vs 51M pilot

---

## Checkpointing & Resume

- `train.py` supports: periodic (~65M tokens / 500 steps), SIGINT/SIGTERM, atomic save/fsync, rotation (`best.pt`, `last.pt`, `step-<N>.pt`)
- RNG state saved for bit-exact resume
- Model config (n_layer, n_head, n_embd, block) configurable via CLI
- `--budget 2500000000` flag works; `--resume` restores full state bit-exactly

---

## Repo Checkpoints (for opencode restarts)

| Milestone | Git Tag |
|-----------|---------|
| Plan finalized | `plan/full-run` ✅ |
| LaTeX stripper ready | `data/latex-stripper` ✅ |
| Re-tokenized corpus | `data/retokenized` ✅ |
| Pretrain started | `train/pretrain-start` ✅ |
| Pretrain 10% (250M) | `train/pretrain-10pct` (pending) |
| Pretrain 50% (1.25B) | `train/pretrain-mid` (pending) |
| Pretrain complete | `train/pretrain-done` (pending) |
| Instruct-tune started | `train/instruct-start` (pending) |
| Instruct-tune complete | `train/instruct-done` (pending) |
| Final eval done | `eval/final` (pending) |

---

## File Changes Completed ✅

1. **`prepare_data.py`** — added `latex-strip` subcommand, `retokenize` subcommand
2. **`train.py`** — model config (n_layer, n_head, n_embd, block) configurable via CLI; `--budget` for 2.5B
3. **`train.py`** — `--resume` restores full state bit-exactly
4. **`sample.py`** — works with checkpoints (needs chat template for instruct phase)
5. **New: `benchmark_vram.py`** — VRAM benchmarking for config selection

---

## Execution Order

1. ✅ Benchmark VRAM (done)
2. ✅ Create plan file (this)
3. ✅ Implement LaTeX stripper in `prepare_data.py`
4. ✅ Run re-tokenization
5. ✅ Update `train.py` for 124M/512 config
6. ✅ Start pretrain (`./run_training.sh --budget 2500000000`)
7. 🔄 Monitor, pause/resume as needed (currently at 207.5M tokens)
8. At 2.5B: run instruction-tune
9. Final evaluation
10. Tag final checkpoints

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OOM at µ-batch 4 | Benchmark confirmed 4.25 GB synthetic; measured 7.8 GB real (near-zero headroom) |
| Throughput degradation | Fresh process restart recovers (CUDA context fragmentation). **Manual monitoring required** — no automated watchdog. |
| LaTeX stripping breaks math | Tested on samples; raw LaTeX preserved in `corpus/clean/` |
| Training instability | 0 AMP skips throughout; GradScaler + clip handles it |
| 2-week wall limit | Checkpoint every ~65M tokens; can pause any time |
| opencode restart | Git tags at milestones; `train.py --resume` handles training resume |

---

## Success Criteria

- Pretrain val loss < 3.0 nats (aggregate) — **currently 4.25 at 207M, on track**
- Instruction-tuned model: coherent multi-sentence STEM responses
- <10% repetition rate at temp 0.8
- EOT token emitted naturally in >50% of completions
- No LaTeX command artifacts in generated text

---

*Plan created 2026-09-09. Updated 2026-09-09 (207.5M tokens, step 1583). Execute phases sequentially.*