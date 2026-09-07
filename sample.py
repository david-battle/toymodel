"""Generate text from a trained checkpoint (plan milestone 5).

Loads a `train.py` checkpoint (`best.pt` weights-only or `last.pt` full
state) and autoregressively samples a continuation from a prompt, with
temperature, top-k and top-p filtering. Runs the model fp16 in eval/no-grad
mode; padded-vocab logits (IDs >= 50257) are masked out as in training, and
generation stops at <|endoftext|>.

Rough-in: written before the pilot run. Recheck sampling quality and options
against a real checkpoint after the pilot, e.g. confirm temperature==0 gives
argmax, that top-p masking retains the top token, and that long prompts
window correctly at BLOCK.

Usage:
  .venv/bin/python sample.py --ckpt ckpt/best.pt \
      --prompt "The electron was discovered in" --max-new-tokens 200 \
      --temperature 0.8 --top-p 0.9
"""

import argparse
import sys

import tiktoken
import torch

from train import BLOCK, DEV, VOCAB_REAL, MiniGPT

EOT = 50256  # GPT-2 <|endoftext|>, also the corpus document separator


def load_model(path):
    """Return (model, enc, meta) from a best.pt (weights-only) or last.pt."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model" not in state:
        sys.exit(f"[sample] {path}: checkpoint has no 'model' key")
    enc = tiktoken.get_encoding("gpt2")
    model = MiniGPT()
    try:
        model.load_state_dict(state["model"])
    except RuntimeError as e:
        sys.exit(f"[sample] {path}: weights do not match the model recipe: {e}")
    model.to(DEV).eval()
    meta = {k: state[k] for k in ("step", "tokens_seen", "best_val")
            if k in state}
    return model, enc, meta


def filter_logits(logits, temperature, top_k, top_p):
    """Return a probability distribution over real vocab after filtering.
    logits: [B, V]. Masked-infinite entries get zero probability."""
    logits = logits[:, :VOCAB_REAL]  # padded-vocab IDs are not real tokens
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = logits.topk(k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sort_logits, sort_idx = logits.sort(dim=-1, descending=True)
        probs = sort_logits.softmax(dim=-1)
        # keep the smallest set of top tokens whose cumulative prob <= top_p
        # (the top token is always kept, so the distribution never vanishes)
        keep = (probs.cumsum(dim=-1) - probs) <= top_p
        sort_logits = sort_logits.masked_fill(~keep, float("-inf"))
        order = sort_idx.argsort(dim=-1)  # un-sort back to original order
        logits = sort_logits.gather(-1, order)
    return logits.softmax(dim=-1)


def sample_one(model, enc, prompt, max_new, temperature, top_k, top_p, gen):
    ctx = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    for _ in range(max_new):
        window = ctx[-BLOCK:]
        x = torch.tensor(window, dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.float16), torch.no_grad():
            logits = model(x)
        last = logits[0, -1:]
        if temperature <= 0.0:
            nxt = int(last.argmax().item())  # greedy
        else:
            probs = filter_logits(last, temperature, top_k, top_p)
            nxt = int(torch.multinomial(probs, 1, generator=gen).item())
        ctx.append(nxt)
        if nxt == EOT:
            break
    return enc.decode(ctx)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="path to best.pt or last.pt")
    ap.add_argument("--prompt", default="",
                    help="starting text (empty = unconditional)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="0 = greedy argmax, otherwise sampling temperature")
    ap.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    ap.add_argument("--top-p", type=float, default=0.9, help="0/1 = disabled")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, enc, meta = load_model(args.ckpt)
    if meta:
        bits = "  ".join(f"{k}={v}" for k, v in meta.items())
        print(f"[sample] loaded {args.ckpt}  ({bits})", flush=True)
    else:
        print(f"[sample] loaded {args.ckpt}", flush=True)

    for i in range(args.num_samples):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(args.seed + i)
        out = sample_one(model, enc, args.prompt, args.max_new_tokens,
                         args.temperature, args.top_k, args.top_p, gen)
        marker = f"[sample {i}] " if args.num_samples > 1 else "[sample] "
        print(f"\n{marker}\n{out}", flush=True)


if __name__ == "__main__":
    main()