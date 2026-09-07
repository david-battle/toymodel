"""Download, curate, tokenize and audit the training corpus.

Layout (all under ./corpus/, gitignored):

  raw/<source>/      downloaded archives/dumps
  clean/<source>.jsonl   one JSON record per kept document (thread/book/paper)
  tokens/<source>.bin    uint16 GPT-2 token stream, <EOT> between documents
  tokens/<source>.offsets.npy  start offsets of each document in the .bin
  manifest.jsonl     per-source metadata + aggregate token counts

Subcommands:
  download <source>...   fetch archives (idempotent)
  extract <source>...    unpack archives (idempotent)
  clean <source>...      normalize archives into clean/<source>.jsonl
  tokenize --source NAME --clean corpus/clean/NAME.jsonl
  audit                  report measured token supply per source

Sources:
  physics, mathematics, chemistry, statistics, cstheory   (StackExchange)
  enwiki, enwikibooks                                      (wikimedia dumps)
  openstax                                                 (textbook chapters)
  arxiv                                                    (RedPajama subset)
"""

import argparse
import html as html_mod
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken

ROOT = Path(__file__).parent.resolve()
CORPUS = ROOT / "corpus"
RAW = CORPUS / "raw"
CLEAN = CORPUS / "clean"
TOKENS = CORPUS / "tokens"
MANIFEST = CORPUS / "manifest.jsonl"

EOT = 50256  # GPT-2 <|endoftext|>
ENC = tiktoken.get_encoding("gpt2")

SE_SITES = ["physics", "math", "chemistry", "stats", "cstheory"]
SE_URL = "https://archive.org/download/stackexchange/{site}.stackexchange.com.7z"
SE_SCORE_MIN = 5

RE_TAG = re.compile(r"<[^>]+>")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# downloads / extraction
# ---------------------------------------------------------------------------

def download_se(sites):
    for site in sites:
        out = RAW / "se" / f"{site}.stackexchange.com.7z"
        if out.exists() and out.stat().st_size > 10_000_000:
            log(f"[se] {site}: already downloaded ({out.stat().st_size/1e9:.2f} GB)")
            continue
        url = SE_URL.format(site=site)
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(out.suffix + ".part")
        log(f"[se] downloading {site} from {url}")
        urllib.request.urlretrieve(url, part)
        part.rename(out)
        log(f"[se] {site}: {out.stat().st_size/1e9:.2f} GB")


def extract_se(sites):
    for site in sites:
        arc = RAW / "se" / f"{site}.stackexchange.com.7z"
        dst = RAW / "se" / site
        posts = dst / "Posts.xml"
        if posts.exists():
            log(f"[se] {site}: already extracted")
            continue
        if not arc.exists():
            sys.exit(f"[se] {site}: archive missing; run download first")
        dst.mkdir(parents=True, exist_ok=True)
        log(f"[se] {site}: extracting ...")
        subprocess.run(["7z", "x", "-y", "-o" + str(dst), str(arc)],
                       stdout=subprocess.DEVNULL, check=True)
        log(f"[se] {site}: extracted")


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

def html_to_text(s):
    """Strip HTML tags/entities; keep code and inline math as plain text."""
    s = html_mod.unescape(s)
    return RE_TAG.sub(" ", s)


def clean_se(site):
    """Convert Posts.xml into one clean/ record per question, keeping the
    accepted-or-top-voted answer thread. Entire threads stay together."""
    import lxml.etree as ET

    posts = RAW / "se" / site / "Posts.xml"
    if not posts.exists():
        sys.exit(f"[se] {site}: Posts.xml missing")
    out = CLEAN / f"se-{site}.jsonl"
    if out.exists():
        log(f"[se] {site}: already cleaned")
        return
    qs = {}  # qid -> (question_attrs, challenge_answers list)
    for _, elem in ET.iterparse(str(posts), tag="row"):
        a = dict(elem.attrib)  # snapshot; elem.clear() below would empty it
        if a.get("PostTypeId") == "1":
            qs[a.get("Id")] = [a, []]
        elif a.get("PostTypeId") == "2" and a.get("ParentId") in qs:
            qs[a.get("ParentId")][1].append(a)
        elem.clear()

    kept = skipped = 0
    with out.open("w", encoding="utf-8") as fh:
        for qid, (q, answers) in qs.items():
            if q.get("DeletionDate"):
                continue
            if float(q.get("Score", 0)) < SE_SCORE_MIN:
                skipped += 1
                continue
            qualified = [a for a in answers
                         if a.get("DeletionDate") is None
                         and float(a.get("Score", 0)) >= SE_SCORE_MIN]
            accepted_id = q.get("AcceptedAnswerId")
            accepted = next((a for a in qualified if a.get("Id") == accepted_id),
                            None)
            sel = accepted or (max(qualified, key=lambda a: float(a.get("Score", 0)))
                               if qualified else None)
            if sel is None:
                skipped += 1
                continue
            title = html_to_text(q.get("Title", "")).strip()
            body = html_to_text(q.get("Body", "")).strip()
            a_body = html_to_text(sel.get("Body", "")).strip()
            if not body or not a_body:
                skipped += 1
                continue
            text = f"# {title}\n\n{body}\n\n## Answer\n\n{a_body}"
            rec = {
                "source": f"se-{site}",
                "id": qid,
                "url": f"https://{site}.stackexchange.com/q/{qid}",
                "date": q.get("CreationDate", ""),
                "score": int(float(q.get("Score", 0))),
                "accepted": accepted is not None,
                "answer_score": int(float(sel.get("Score", 0))),
                "text": text,
            }
            fh.write(json.dumps(rec) + "\n")
            kept += 1
    log(f"[se] {site}: kept {kept} threads, skipped {skipped}")


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def tokenize(source, clean_file):
    out_bin = TOKENS / f"{source}.bin"
    out_off = TOKENS / f"{source}.offsets.npy"
    if out_bin.exists():
        log(f"[tok] {source}: already tokenized")
        return
    log(f"[tok] {source}: tokenizing {clean_file}")
    offsets = [0]
    total = 0
    with open(clean_file, encoding="utf-8") as fh, out_bin.open("wb") as fout:
        for line in fh:
            rec = json.loads(line)
            ids = ENC.encode(rec["text"]) + [EOT]
            fout.write(np.asarray(ids, dtype=np.uint16).tobytes())
            total += len(ids)
            offsets.append(total)
    np.save(out_off, np.asarray(offsets, dtype=np.int64))
    docs = len(offsets) - 1
    log(f"[tok] {source}: docs={docs} tokens={total/1e6:.1f}M -> {out_bin}")
    with MANIFEST.open("a", encoding="utf-8") as m:
        m.write(json.dumps({"source": source, "docs": docs, "tokens": total}) + "\n")


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit():
    rows = []
    if MANIFEST.exists():
        for manifest_line in MANIFEST.open(encoding="utf-8"):
            rows.append(json.loads(manifest_line))
    have = {m["source"] for m in rows}
    for bin_path in sorted(TOKENS.glob("*.bin")):
        src = bin_path.stem
        if src in have:
            continue
        off = np.load(TOKENS / f"{src}.offsets.npy")
        rows.append({"source": src, "docs": len(off) - 1, "tokens": int(off[-1])})
    rows.sort(key=lambda m: m["source"])
    print(f"{'source':<16}{'docs':>10}{'tokens':>14}{'tokens/M':>10}{'MB':>9}")
    tot_docs = tot_tok = 0
    for m in rows:
        off = np.load(TOKENS / f"{m['source']}.offsets.npy")
        b = (TOKENS / f"{m['source']}.bin").stat().st_size
        tot_docs += len(off) - 1
        tot_tok += m["tokens"]
        print(f"{m['source']:<16}{len(off)-1:>10}{m['tokens']:>14}{m['tokens']/1e6:>10.1f}{b/1e6:>9.1f}")
    print(f"{'TOTAL':<16}{tot_docs:>10}{tot_tok:>14}{tot_tok/1e6:>10.1f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="download sources (idempotent)")
    p.add_argument("source", nargs="+", choices=["se"])
    p.add_argument("--sites", nargs="*", default=SE_SITES)

    p = sub.add_parser("extract", help="extract archives")
    p.add_argument("source", nargs="+", choices=["se"])
    p.add_argument("--sites", nargs="*", default=SE_SITES)

    p = sub.add_parser("clean", help="normalize archives to clean/")
    p.add_argument("source", nargs="+", choices=["se"])
    p.add_argument("--sites", nargs="*", default=SE_SITES)

    p = sub.add_parser("tokenize", help="tokenize a clean jsonl into tokens/")
    p.add_argument("--source", required=True)
    p.add_argument("--clean", required=True)

    sub.add_parser("audit", help="show token supply per source")

    args = ap.parse_args()
    if args.cmd == "download":
        for _ in args.source:
            download_se(args.sites)
    elif args.cmd == "extract":
        for _ in args.source:
            extract_se(args.sites)
    elif args.cmd == "clean":
        for _ in args.source:
            for site in args.sites:
                clean_se(site)
    elif args.cmd == "tokenize":
        tokenize(args.source, args.clean)
    elif args.cmd == "audit":
        audit()


if __name__ == "__main__":
    main()