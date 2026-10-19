#!/usr/bin/env python3
"""Check that prefix-cache hits reproduce uncached outputs, on one live server.

Every probe prompt is answered twice, greedily:
  hit - a normal request, which reads the prefix cache
  ref - the same request with prompt_logprobs=0, which makes vLLM skip reading
        the prefix cache (SamplingParams.skip_reading_prefix_cache)
so the reference comes from the same server, weights and kernels, and no
separate cache-off server is needed.

This is a smoke test of the target model's cache-hit path; the thorough
cache-vs-no-cache checks live in vLLM's own tests (test_hybrid.py -k apc,
v1/e2e/general/test_mamba_prefix_cache.py), which run on small models.
By default hits are checked one request at a time (seq). Two deeper phases are
opt-in, both judged against the same sequential references:
  --concurrent - every hit at once over the warm cache, so cached-context
                 prefills share batches with each other, as they do under load
  --evict      - flood the cache with unrelated prompts until the probe's
                 blocks are evicted, then ask again (evicted: the recompute must
                 be right) and once more (rehit: the refilled blocks must be
                 right too). Skipped when the KV pool is too big to flood
                 cheaply (--evict-max-tokens).
The default prefix lengths are one block-aligned and one not (4096, 4001), so
hits end both exactly at the shared prefix and inside it, where the tail of the
shared prefix has to be recomputed. Not covered: preemption/resume, which needs
memory pressure mid-request and cannot be triggered reliably from outside.

Prompts are natural text (vLLM's own docs, cut into fixed-length prefixes) plus
short questions, so the model is confident and greedy decoding is stable.
Random-token prompts are useless here: their logits are near-flat and a 1e-3
numeric wobble flips the first token even with plain attention.

A divergence is judged at the first differing token, as vLLM's own
check_logprobs_close (tests/models/utils.py) does:
  close    - each side's token is in the other's top-TOP_K logprobs, i.e. both
             runs saw it as a likely candidate and numeric noise tipped greedy
  MISMATCH - either token is missing from the other side's top-TOP_K
margin (the reference's top-1 logprob minus the hit token's) and first_dlp are
kept for diagnosis only; they do not affect the verdict.
first_dlp is |logprob(hit) - logprob(ref)| of the reference's first token: the
first decode step reads the restored cache state directly, so it is the most
direct measure of whether that state is right. Healthy runs stay well under
0.1; a wrong state shows up as >1 and a divergence at token 0.

Server must run with --enable-prompt-tokens-details so each request reports
how many prompt tokens it actually read from the cache.

Usage: python3 pc_accuracy.py --model M [--port 9999] [--out accuracy.json]
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

QUESTIONS = [
    "\n\nQuestion: In one sentence, what is the text above mainly about?\nAnswer:",
    "\n\nQuestion: Name three technical terms that appear in the text above.\nAnswer:",
    "\n\nQuestion: Who is the intended reader of the text above?\nAnswer:",
    "\n\nQuestion: Suggest a short title for the text above.\nAnswer:",
]
# Logprobs per step; vLLM's tests use 5 (num_logprobs) with check_logprobs_close.
TOP_K = 5


def post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return r.read().decode()


def tokenize(base, model, text, special):
    return post(base, "/tokenize", {"model": model, "prompt": text,
                                    "add_special_tokens": special})["tokens"]


def md_files(corpus):
    return sorted(glob.glob(os.path.join(corpus, "**", "*.md"), recursive=True))


def build_prompts(base, model, corpus, lengths, num_prefixes):
    """[(length, prefix tokens)], the corpus tokens and the question tokens."""
    files = md_files(corpus)
    if not files:
        sys.exit(f"no .md files under {corpus}; point --corpus (PROBE_CORPUS in "
                 "bench_prefix_caching.sh) at any folder of .md prose")
    text = "\n\n".join(open(f, errors="ignore").read() for f in files)
    ids = tokenize(base, model, text, special=False)
    windows = len(lengths) * num_prefixes
    # Non-overlapping windows spread across the corpus, so prefixes are distinct.
    stride = len(ids) // windows
    if stride < max(lengths):
        sys.exit(f"corpus has {len(ids)} tokens, need {windows * max(lengths)}")
    bos = tokenize(base, model, "", special=True)  # [] or [bos], per tokenizer
    questions = [tokenize(base, model, q, special=False) for q in QUESTIONS]
    # The BOS counts toward the length, so 4096 stays block-aligned with it.
    prompts = [(length, bos + ids[w * stride:w * stride + length - len(bos)])
               for w, length in enumerate(l for l in lengths for _ in range(num_prefixes))]
    return prompts, ids, questions


def complete(base, model, prompt, max_tokens, skip_cache):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "ignore_eos": True, "logprobs": TOP_K,
               "return_tokens_as_token_ids": True}
    if skip_cache:
        payload["prompt_logprobs"] = 0
    r = post(base, "/v1/completions", payload)
    lp = r["choices"][0]["logprobs"]
    details = (r.get("usage") or {}).get("prompt_tokens_details") or {}
    return {"text": r["choices"][0]["text"], "tokens": lp["tokens"],
            "top": lp["top_logprobs"], "cached": details.get("cached_tokens")}


def classify(hit, ref):
    """(verdict, diverge_at, first_dlp, margin, hit_top, ref_top); the top-k
    dicts are those at the divergence, kept so a verdict can be rechecked."""
    t0 = ref["tokens"][0]
    first_dlp = abs(hit["top"][0].get(t0, -math.inf) - ref["top"][0][t0])
    k = next((i for i, (a, b) in enumerate(zip(hit["tokens"], ref["tokens"]))
              if a != b), None)
    if k is None:
        return "same", None, first_dlp, None, None, None
    hit_top, ref_top = hit["top"][k], ref["top"][k]
    hit_tok, ref_tok = hit["tokens"][k], ref["tokens"][k]
    margin = ref_top[ref_tok] - ref_top.get(hit_tok, -math.inf)
    close = hit_tok in ref_top and ref_tok in hit_top
    return ("close" if close else "MISMATCH"), k, first_dlp, margin, hit_top, ref_top


def kv_capacity(base):
    """Tokens the KV pool holds, from vllm:cache_config_info; None if unknown."""
    try:
        m = re.search(r"^vllm:cache_config_info\{(.*)\}", get(base, "/metrics"), re.M)
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1))) if m else {}
        return int(labels["num_gpu_blocks"]) * int(labels["block_size"])
    except (OSError, KeyError, ValueError):
        return None


def evict_phase(base, args, prompts, questions, ids, check):
    """Flood the cache past its size, then re-ask one sibling per prefix twice.
    Returns a one-line outcome for the summary."""
    cap = kv_capacity(base)
    if cap is None:
        return "skipped (KV size unknown)"
    # 1.5x: LRU evicts the probe's (oldest) blocks first, with margin for
    # blocks the flood itself frees and reuses.
    flood_tokens = int(1.5 * cap)
    if flood_tokens > args.evict_max_tokens:
        return f"skipped (KV {cap} tok > --evict-max-tokens / 1.5)"
    max_len = json.loads(get(base, "/v1/models"))["data"][0]["max_model_len"]
    flood_len = min(4096, max_len - 64)
    span = len(ids) - flood_len
    if span <= 0:
        return "skipped (corpus shorter than one flood prompt)"

    def flood(i):
        # A distinct opening makes every block of every flood prompt distinct.
        head = tokenize(base, args.model, f"Flood document {i}.\n", special=False)
        start = (i * 7919) % span
        complete(base, args.model, head + ids[start:start + flood_len - len(head)], 1, False)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(flood, range(-(-flood_tokens // flood_len))))

    qi = 2  # a sibling question: reused only through the shared prefix
    evicted = 0
    for p, (length, prefix) in enumerate(prompts):
        prompt = prefix + questions[qi]
        first = complete(base, args.model, prompt, args.max_tokens, skip_cache=False)
        evicted += (first["cached"] or 0) < length // 2
        check(p, qi, "evicted", first)
        check(p, qi, "rehit", complete(base, args.model, prompt, args.max_tokens, skip_cache=False))
    return f"{evicted}/{len(prompts)} evicted"


def summarize(results, evict, dlp_warn):
    seq = [r for r in results if r["phase"] == "seq"]
    known = [r for r in seq if r["cached_tokens"] is not None]
    hits = [r for r in known if r["cached_tokens"] > 0]
    # Judge every answer that read the cache, plus the post-eviction recomputes.
    # No hits at all (e.g. caching off): judge everything instead, as a
    # self-check that the probe's two paths agree when neither reads the cache.
    tested = [r for r in results if r["cached_tokens"] or r["phase"] == "evicted"]
    tested = tested if hits else results
    count = {v: sum(r["verdict"] == v for r in tested) for v in ("same", "close", "MISMATCH")}
    max_dlp = max((r["first_dlp"] for r in tested), default=0.0)
    siblings = [r for r in known if not r["repeat"]]
    sibling_hits = [r for r in siblings if r["cached_tokens"] > 0]
    if any(r["ref_cached_tokens"] for r in results):
        verdict = "BAD-REF"  # prompt_logprobs no longer bypasses the cache
    elif known and not hits:
        verdict = "NO-HIT"
    elif count["MISMATCH"]:
        verdict = "FAIL"
    elif max_dlp > dlp_warn:
        verdict = "WARN"
    elif known and not sibling_hits:
        # Outputs are right, but only verbatim repeats hit: shared-prefix reuse
        # (what this benchmark measures) never happened.
        verdict = "NO-SIBLING-HIT"
    else:
        verdict = "PASS"
    if known:
        repeat_hits = len(hits) - len(sibling_hits)
        hit_info = (f"hit sibling {len(sibling_hits)}/{len(siblings)} "
                    f"repeat {repeat_hits}/{len(known) - len(siblings)}")
    else:
        hit_info = "hit ? (start server with --enable-prompt-tokens-details)"
    scope = "self-check " if known and not hits else ""
    conc = [r for r in results if r["phase"] == "conc"]
    phases = f"  conc {sum(r['verdict'] != 'MISMATCH' for r in conc)}/{len(conc)} ok" if conc else ""
    phases += f"  evict: {evict}" if evict else ""
    return verdict, (f"{verdict}  {hit_info}  {scope}same {count['same']} close {count['close']} "
                     f"mismatch {count['MISMATCH']}  max-first-dlp {max_dlp:.3f}{phases}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    # Any folder of .md prose works; hit and ref share it, so which copy of the
    # docs is used does not matter. Prefer the container's vLLM checkout.
    # A pip-installed vLLM can leave /workspace/vllm without docs, so pick the
    # first candidate that actually holds .md files.
    corpora = ["/workspace/vllm/docs", os.path.join(here, "workspace", "vllm", "docs")]
    ap.add_argument("--corpus", default=next((c for c in corpora if md_files(c)), corpora[0]))
    ap.add_argument("--prefix-lens", type=int, nargs="+", default=[4096, 4001],
                    help="shared-prefix lengths; keep one that no block size divides")
    ap.add_argument("--num-prefixes", type=int, default=3, help="distinct prefixes per length")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--dlp-warn", type=float, default=0.25,
                    help="warn when the first token's logprob moves more than this")
    ap.add_argument("--concurrent", action="store_true", help="also run the concurrent phase")
    ap.add_argument("--evict", action="store_true", help="also run the eviction phase")
    ap.add_argument("--evict-max-tokens", type=int, default=2_000_000,
                    help="largest flood (tokens) the eviction phase may send")
    ap.add_argument("--out", help="write per-request results here (json)")
    args = ap.parse_args()

    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    base = f"http://{args.host}:{args.port}"

    prompts, ids, questions = build_prompts(base, args.model, args.corpus,
                                            args.prefix_lens, args.num_prefixes)
    results, refs = [], {}

    def check(p, qi, phase, hit):
        length, prefix = prompts[p]
        ref = refs[p, qi]
        verdict, div_at, first_dlp, margin, hit_top, ref_top = classify(hit, ref)
        results.append({"prefix": p, "length": length, "question": qi, "phase": phase,
                        "prompt_len": len(prefix) + len(questions[qi]),
                        # q0/q1 repeat a warm-up prompt verbatim (replay-boundary
                        # reuse); the rest only share the prefix (sibling reuse).
                        "repeat": qi < 2,
                        "cached_tokens": hit["cached"], "ref_cached_tokens": ref["cached"],
                        "verdict": verdict, "diverge_at": div_at,
                        "first_dlp": first_dlp, "margin": margin,
                        "hit_top": hit_top, "ref_top": ref_top,
                        "hit_text": hit["text"], "ref_text": ref["text"]})

    for p, (length, prefix) in enumerate(prompts):
        # Two warm-up siblings: the first fills the cache; the second lets
        # sparse retention (prefix_cache_retention_interval=0) detect the
        # shared-prefix junction and checkpoint it, so later siblings can hit.
        for q in questions[:2]:
            complete(base, args.model, prefix + q, args.max_tokens, skip_cache=False)
        for qi, q in enumerate(questions):
            hit = complete(base, args.model, prefix + q, args.max_tokens, skip_cache=False)
            refs[p, qi] = complete(base, args.model, prefix + q, args.max_tokens, skip_cache=True)
            check(p, qi, "seq", hit)

    if args.concurrent:
        jobs = [(p, qi) for p in range(len(prompts)) for qi in range(len(questions))]
        with ThreadPoolExecutor(max_workers=min(32, len(jobs))) as ex:
            hits = list(ex.map(lambda j: complete(base, args.model, prompts[j[0]][1] + questions[j[1]],
                                                  args.max_tokens, skip_cache=False), jobs))
        for (p, qi), hit in zip(jobs, hits):
            check(p, qi, "conc", hit)

    evict = evict_phase(base, args, prompts, questions, ids, check) if args.evict else None

    verdict, summary = summarize(results, evict, args.dlp_warn)
    print(summary)
    for r in results:
        if r["verdict"] == "MISMATCH":
            print(f"  [{r['phase']}] len {r['length']} prefix {r['prefix']} q{r['question']} "
                  f"@tok {r['diverge_at']} (cached {r['cached_tokens']}/{r['prompt_len']})\n"
                  f"    hit: {r['hit_text'][:100]!r}\n    ref: {r['ref_text'][:100]!r}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "summary": summary, "verdict": verdict,
                       "evict": evict, "requests": results}, f, indent=1)
    return 0 if verdict in ("PASS", "WARN", "NO-SIBLING-HIT") else 1


if __name__ == "__main__":
    sys.exit(main())
