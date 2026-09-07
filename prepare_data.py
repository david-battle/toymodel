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
import io
import json
import re
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
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
# arXiv: ranked-direct-from-arXiv (OpenAlex ranking + ar5iv/arxiv HTML)
# ---------------------------------------------------------------------------

OPENALEX_ARXIV = "S4306400194"
ARXIV_ALLOWED = ("physics.", "math.", "cs.", "cond-mat", "astro-ph", "quant-ph",
                 "stat.", "nlin.")
ARXIV_RANK_MAX = 200


def _arxiv_id_from_url(url):
    if not url:
        return None
    if "arxiv.org/abs/" not in url:
        return None
    aid = url.split("arxiv.org/abs/", 1)[1]
    return re.sub(r"v[0-9]+$", "", aid).strip()


def arxiv_rank(top, mailto):
    """Rank arXiv works by citations (OpenAlex), annotate arXiv categories,
    keep allowed STEM categories, write corpus/raw/arxiv/ranked.jsonl."""
    import requests
    candidates = top * 2  # fetch margin for category filtering
    works = []
    cursor = "*"
    got = 0
    while got < candidates:
        params = {
            "filter": (f"locations.source.id:{OPENALEX_ARXIV},"
                       "publication_year:2000-2026"),
            "sort": "cited_by_count:desc",
            "per-page": ARXIV_RANK_MAX,
            "cursor": cursor,
            "mailto": mailto,
        }
        r = requests.get("https://api.openalex.org/works", params=params,
                         timeout=60)
        r.raise_for_status()
        data = r.json()
        batch = data["results"]
        if not batch:
            break
        for w in batch:
            aid = None
            for loc in w.get("locations", []):
                aid = _arxiv_id_from_url((loc.get("landing_page_url") or ""))
                if aid:
                    break
            if aid:
                works.append({"id": aid, "cited": w.get("cited_by_count", 0)})
        got += len(batch)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        if got % 2000 < ARXIV_RANK_MAX:
            log(f"[arxiv] OpenAlex: {got} works seen")

    # dedupe, keep union order
    seen, ordered = set(), []
    for w in works:
        if w["id"] not in seen:
            seen.add(w["id"])
            ordered.append(w)

    # batch-annotate arXiv categories via the arXiv API (Atom feed)
    ATOM = {"a": "http://www.w3.org/2005/Atom",
            "ax": "http://arxiv.org/schemas/atom"}
    cats = {}
    chunk = 300
    for i in range(0, len(ordered), chunk):
        ids = ",".join(w["id"] for w in ordered[i:i + chunk])
        url = (f"http://export.arxiv.org/api/query?id_list={ids}"
               "&max_results=1000")
        feed = None
        for attempt in range(3):
            try:
                feed = requests.get(url, timeout=120).text
                break
            except Exception:
                time.sleep(5)
        try:
            tree = ET.parse(io.StringIO(feed))
            for entry in tree.getroot().findall("a:entry", ATOM):
                eid = entry.findtext("a:id", default="", namespaces=ATOM)
                aid = re.sub(r"v[0-9]+$", "", eid.split("/")[-1])
                pc = entry.find("ax:primary_category", ATOM)
                cats[aid] = pc.get("term") if pc is not None else None
        except Exception as exc:
            log(f"[arxiv] category batch {i//chunk} failed: {exc}")
        if i // chunk and i % (chunk * 5) == 0:
            log(f"[arxiv] categories: {len(cats)}/{len(ordered)}")
        time.sleep(3)

    ranked = []
    dropped = 0
    for w in ordered:
        cat = cats.get(w["id"])
        if not cat or not cat.startswith(ARXIV_ALLOWED):
            dropped += 1
            continue
        w["category"] = cat
        ranked.append(w)
    out_dir = RAW / "arxiv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ranked.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for w in ranked:
            fh.write(json.dumps(w) + "\n")
    log(f"[arxiv] ranked {len(ranked)} papers (dropped {dropped} off-category)"
        f" -> {out}")

END_MARKERS = ("\nreferences\n", "\nbibliography\n", "\nacknowledgments\n",
               "\nacknowledgements\n", "\nappendix\n")


def _html_to_text(html_bytes):
    import lxml.html as lh
    root = lh.fromstring(html_bytes)
    for tag in ("script", "style", "nav", "header", "footer"):
        for el in root.xpath(f"//{tag}"):
            el.drop_tree()
    for el in root.xpath("//main | //article | //*[@class='ltx_page_main']"):
        root = el
        break
    text = " ".join((root.text_content() or "").split())
    lower = text.lower()
    for marker in END_MARKERS:
        idx = lower.find(marker)
        if idx >= 0:
            text = text[:idx]
            break
    return text


def arxiv_fetch(max_tokens, workers):
    """Fetch ranked arXiv papers as HTML (arXiv HTML5, else ar5iv), convert to
    plain text, append to clean/arxiv.jsonl, stop once quota met."""
    import concurrent.futures as cf

    ranked_file = RAW / "arxiv" / "ranked.jsonl"
    if not ranked_file.exists():
        sys.exit("[arxiv] run `arxiv rank` first")
    papers = [json.loads(l) for l in ranked_file.open(encoding="utf-8")]

    done_ids = set()
    out = CLEAN / "arxiv.jsonl"
    if out.exists():
        for line in out.open(encoding="utf-8"):
            done_ids.add(json.loads(line)["id"])
    progress = RAW / "arxiv" / "fetched.log"
    seen_done = set()
    if progress.exists():
        seen_done = {l.strip() for l in progress.open()}

    pending = [p for p in papers if p["id"] not in done_ids
               and p["id"] not in seen_done]
    by_cat = {}
    for p in pending:
        by_cat.setdefault(p["category"], []).append(p)
    # round-robin across categories so the citation quota samples every field
    pending = []
    buckets = [list(b) for b in by_cat.values()]
    while any(buckets):
        for b in buckets:
            if b:
                pending.append(b.pop(0))
    log(f"[arxiv] {len(done_ids)} done, {len(pending)} to fetch "
        f"(quota {max_tokens/1e6:.0f}M clean tokens)")

    def fetch_one(paper):
        aid = paper["id"]
        html = None
        for base in (f"https://arxiv.org/html/{aid}",
                     f"https://ar5iv.labs.arxiv.org/html/{aid}"):
            for attempt in range(2):
                try:
                    r = requests.get(base, timeout=45, headers={
                        "User-Agent": "toymodel-curator/0.1"})
                    if r.status_code == 200 and len(r.content) > 50_000:
                        html = r.content
                        break
                except Exception:
                    pass
                time.sleep(1.5)
            if html is not None:
                break
        if html is None:
            return None
        text = _html_to_text(html)
        if len(text) < 2000:
            return None
        return {"id": aid, "category": paper.get("category"),
                "title": paper.get("title"), "text": text}

    kept = 0
    tok_est = 0
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, p): p for p in pending}
        try:
            for fut in cf.as_completed(futs):
                rec = fut.result()
                p = futs[fut]
                progress.open("a").write(p["id"] + "\n")
                if rec is None:
                    continue
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n = len(rec["text"]) // 4
                kept += 1
                tok_est += n
                if kept % 50 == 0:
                    el = time.time() - start
                    log(f"[arxiv] kept {kept} papers, ~{tok_est/1e6:.0f}M tok, "
                        f"{el/60:.1f} min")
                if tok_est >= max_tokens:
                    log("[arxiv] quota reached, stopping")
                    for fx in futs:
                        fx.cancel()
                    break
        finally:
            for fx in futs:
                fx.cancel()
    log(f"[arxiv] fetched {kept} papers (~{tok_est/1e6:.0f}M tokens est)")

# ---------------------------------------------------------------------------
# Wikipedia: stream hf wikimedia/wikipedia, keep STEM-titled articles only
# ---------------------------------------------------------------------------

STEM_RE = re.compile(
    r"\b(physics|physical|mathematics|mathematical|math\b|calculus|algebra|"
    r"geometry|topology|statistics|statistical|probability|probabilistic|"
    r"analysis\b|differential|integral\b|equation|theorem|theory|function|"
    r"algorithm|computational|complexity|programming|computing|neural|"
    r"machine learning|quantum|particle|electromagnet|thermodynamics|"
    r"mechanics|relativity|astronom|cosmolog|chemistry|chemical|molecule|"
    r"atomic|protein|genome|biology|evolution|ecology|genetics|neur|tensor|"
    r"graph\b|number theory|logic|cryptograph|entropy|symmetry|topolog)\b",
    re.I)
SKIP_RE = re.compile(
    r"(disambiguation|^list of|\(film\)|\(album\)|\(song\)|\(novel\)|"
    r"^wikipedia:|^template:|^category:)", re.I)


def _wiki_clean(text):
    text = " ".join(text.split())
    lower = text.lower()
    for marker in END_MARKERS:
        idx = lower.find(marker)
        if idx >= 0:
            text = text[:idx]
            break
    return text


def wiki_fetch(max_tokens, shards):
    """Stream the hf en Wikipedia partition, keep only STEM-titled articles,
    append to clean/wikipedia.jsonl, stop once the token quota is met."""
    from datasets import load_dataset

    out = CLEAN / "wikipedia.jsonl"
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            done.add(json.loads(line)["id"])
    log(f"[wiki] {len(done)} articles already kept")

    kept = 0
    tok = 0
    seen = 0
    shard_limit = shards  # None or int of max shards to scan
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                      streaming=True)
    with out.open("a", encoding="utf-8") as fh:
        for row in ds:
            cur_shard = int(row.get("id", "0").split("-")[0])
            if shard_limit is not None and cur_shard >= shard_limit:
                break
            if row["id"] in done:
                continue
            seen += 1
            title = row.get("title") or ""
            if SKIP_RE.search(title) or not STEM_RE.search(title):
                continue
            text = _wiki_clean(row.get("text") or "")
            if len(text) < 500:
                continue
            fh.write(json.dumps({
                "id": row["id"], "url": row.get("url", ""),
                "title": title, "text": text}, ensure_ascii=False) + "\n")
            kept += 1
            tok += len(text) // 4
            if kept % 1000 == 0:
                log(f"[wiki] seen {seen}, kept {kept} (~{tok/1e6:.0f}M tok)"
                    f" shard {cur_shard}")
            if tok >= max_tokens:
                log("[wiki] quota reached, stopping")
                break
    log(f"[wiki] kept {kept} STEM-titled articles (~{tok/1e6:.0f}M tokens est)")


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

    p = sub.add_parser("arxiv-rank", help="rank arXiv by citations, keep STEM")
    p.add_argument("--top", type=int, default=4000)
    p.add_argument("--mailto", default="dlbattle@example.com",
                   help="email for OpenAlex polite pool")

    p = sub.add_parser("arxiv-fetch", help="fetch ranked papers as clean text")
    p.add_argument("--max-tokens", type=int, default=120_000_000)
    p.add_argument("--workers", type=int, default=6)

    p = sub.add_parser("wiki-fetch", help="stream Wikipedia, keep STEM titles")
    p.add_argument("--max-tokens", type=int, default=50_000_000)
    p.add_argument("--shards", type=int, default=None,
                   help="stop after N shards (testing)")

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
    elif args.cmd == "arxiv-rank":
        arxiv_rank(args.top, args.mailto)
    elif args.cmd == "arxiv-fetch":
        arxiv_fetch(args.max_tokens, args.workers)
    elif args.cmd == "wiki-fetch":
        wiki_fetch(args.max_tokens, args.shards)
    elif args.cmd == "audit":
        audit()


if __name__ == "__main__":
    main()