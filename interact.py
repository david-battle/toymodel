"""Interactive one-line text completion against a trained checkpoint.

Loads the model once, then reads one line at a time from stdin and prints a
single-line continuation of up to CHAR_BUDGET chars, stopping early if the
model emits an <|endoftext|> token or a newline. A blank line generates an
unconditional continuation; `quit`/`exit` or Ctrl-D ends.

Usage:
  .venv/bin/python interact.py [--ckpt ckpt/best.pt]
"""

import argparse
import sys

import torch

from sample import EOT, filter_logits, load_model
from train import BLOCK, DEV

CHAR_BUDGET = 80
HEADROOM = 32  # tokens reserved for the response when the prompt fills the window
BLANK_SEEDS = ("The", "A")  # a continuation model has no BOS; blank lines get an opener
RETRIES = 5
GIVE_UP_EMPTY = 16  # leading tokens still blank after this => model won't escape


def _generate(model, enc, ctx, temperature, top_k, top_p):
    gen0 = len(ctx)
    for i in range(BLOCK - gen0):
        window = ctx[-BLOCK:]
        x = torch.tensor(window, dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.float16), torch.no_grad():
            logits = model(x)
        last = logits[0, -1:]
        if temperature <= 0.0:
            nxt = int(last.argmax().item())
        else:
            probs = filter_logits(last, temperature, top_k, top_p)
            nxt = int(torch.multinomial(probs, 1).item())
        ctx.append(nxt)
        eff = enc.decode(ctx[gen0:]).lstrip()
        if nxt == EOT and eff:
            break
        if eff and ("\n" in eff or len(eff) >= CHAR_BUDGET):
            break
        if not eff and i >= GIVE_UP_EMPTY:
            return ""  # stuck emitting <|endoftext|>/whitespace; caller retries
    return enc.decode(ctx[gen0:]).lstrip().split("\n")[0][:CHAR_BUDGET]


def complete_line(model, enc, prompt, temperature, top_k, top_p):
    if not prompt.strip():
        for seed in BLANK_SEEDS:
            for _ in range(RETRIES):
                out = _generate(model, enc, enc.encode(seed), temperature, top_k, top_p)
                if out:
                    return out
        return ""
    ctx = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    ctx = ctx[-(BLOCK - HEADROOM):]  # long prompts keep only the recent tail
    for _ in range(RETRIES):
        out = _generate(model, enc, list(ctx), temperature, top_k, top_p)
        if out:
            return out
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/best.pt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    ap.add_argument("--top-p", type=float, default=0.9, help="0/1 = disabled")
    args = ap.parse_args()

    model, enc, _ = load_model(args.ckpt)
    print(f"[interact] loaded {args.ckpt}  (type a line, `quit`/`exit` or Ctrl-D to stop)",
          flush=True)
    while True:
        try:
            prompt = input(">>> ")
        except EOFError:
            print()
            break
        if not prompt.strip():
            prompt = ""  # blank line = unconditional continuation
        elif prompt.strip().lower() in ("quit", "exit"):
            print("bye")
            break
        out = complete_line(model, enc, prompt, args.temperature,
                            args.top_k, args.top_p)
        print(out, flush=True)


if __name__ == "__main__":
    main()