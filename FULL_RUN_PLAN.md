# Full Run Plan: 124M GPT on 2.5B Tokens

**Target**: 124M-param decoder-only GPT (L12 H12 D768), context 512, trained on 2.5B tokens of cleaned STEM corpus with instruction-tuning phase.

**Hardware**: RTX 2080 Super Max-Q 8GB (≈6.5 GB usable after display)

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
| Eval interval | 50M tokens |
| Checkpoint interval | 25M tokens + signals |

**VRAM**: ~4.3 GB peak allocated (benchmarked), safe with 1 GB display overhead.

**Throughput**: ~9,700 tok/s sustained (benchmarked).

**Time estimate**: 2.5B / 9,700 ≈ **71.5 hours (3 days)** pretrain + 4h instruct-tune = **~3.5 days GPU time**. 2-week wall budget allows for pauses, data prep, eval.

---

## Data Pipeline

### Phase 1: LaTeX Stripping & Re-tokenization (2–4h)
- Parse `corpus/clean/*.jsonl` with `pylatexenc` + custom heuristics
- Convert LaTeX math → Unicode/natural language:
  - Inline `$...$` → plain text with Unicode symbols (∑, ∫, ∈, etc.)
  - Display `$$...$$` → same, on own line
  - `\frac{a}{b}` → `a/b`, `\sqrt{x}` → `√x`, `\mathbb{R}` → `ℝ`
  - Greek letters, arrows, relations → Unicode
  - Preserve plain text structure
- Strip `## Answer` markers from SE data
- Re-tokenize with tiktoken GPT-2 → `corpus/tokens_v2/*.bin`
- Audit token counts, update manifest

### Phase 2: Pretraining (72h GPU)
- 2.5B token budget (9.3 epochs over 269M unique tokens)
- Mixture: arXiv 60% / SE 28% / Wikipedia 12%
- Held-out: 0.5% per source (fixed split, same as pilot)
- Resume from `ckpt/last.pt` on interrupt

### Phase 3: Instruction Tuning (4h GPU)
- Format SE Q&A as chat: `Q: <question>\nA: <accepted_answer><|endoftext|>`
- Filter: score ≥ 5, accepted answer exists
- ~100k examples × 3 epochs = ~300M tokens
- LR: 1e-4 constant (no decay), µ-batch 8, accum 32
- Start from best pretrain checkpoint

### Phase 4: Evaluation (1h)
- Perplexity on held-out test sets (per source + aggregate)
- Generation quality: fixed prompts, temperature sweep
- Coherence metrics: repetition rate, EOT emission, LaTeX artifacts
- Comparison vs 51M pilot

---

## Checkpointing & Resume

- `train.py` already supports: periodic (25M tokens), SIGINT/SIGTERM, atomic save/fsync, rotation (`best.pt`, `last.pt`, `step-<N>.pt`)
- RNG state saved for bit-exact resume
- Add `--budget 2500000000` flag
- Add `--context 512` flag (or config)

---

## Repo Checkpoints (for opencode restarts)

| Milestone | Git Tag |
|-----------|---------|
| Plan finalized | `plan/full-run` |
| LaTeX stripper ready | `data/latex-stripper` |
| Re-tokenized corpus | `data/retokenized` |
| Pretrain started | `train/pretrain-start` |
| Pretrain 50% (1.25B) | `train/pretrain-mid` |
| Pretrain complete | `train/pretrain-done` |
| Instruct-tune started | `train/instruct-start` |
| Instruct-tune complete | `train/instruct-done` |
| Final eval done | `eval/final` |

---

## File Changes Needed

1. **`prepare_data.py`** — add `latex-strip` subcommand, `retokenize` subcommand
2. **`train.py`** — make `N_LAYER`, `N_HEAD`, `N_EMBD`, `BLOCK` configurable via CLI; add `--budget` for 2.5B
3. **`train.py`** — add `--mode instruct` for instruction-tuning phase
4. **`sample.py`** — support chat template, temperature sweep eval
5. **New: `eval.py`** — automated evaluation script

---

## Execution Order

1. ✅ Benchmark VRAM (done)
2. Create plan file (this)
3. Implement LaTeX stripper in `prepare_data.py`
4. Run re-tokenization
5. Update `train.py` for 124M/512 config
6. Start pretrain (`./run_training.sh --budget 2500000000`)
7. Monitor, pause/resume as needed
8. At 2.5B: run instruction-tune
9. Final evaluation
10. Tag final checkpoints

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OOM at µ-batch 4 | Benchmark confirms 4.25 GB; fallback to µ-batch 2 (8.7k tok/s, 80h) |
| LaTeX stripping breaks math | Test on samples first; keep raw LaTeX as fallback corpus |
| Training instability | 0 AMP skips in pilot; GradScaler + clip handles it |
| 2-week wall limit | Checkpoint every 25M tokens; can pause any time |
| opencode restart | Git tags at each milestone; `train.py --resume` handles training resume |

---

## Success Criteria

- Pretrain val loss < 3.0 nats (aggregate)
- Instruction-tuned model: coherent multi-sentence STEM responses
- <10% repetition rate at temp 0.8
- EOT token emitted naturally in >50% of completions
- No LaTeX command artifacts in generated text

---

*Plan created 2026-09-09. Execute phases sequentially.*