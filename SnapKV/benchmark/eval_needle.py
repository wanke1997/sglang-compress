"""SnapKV long-context needle-in-a-haystack validation.

Reproduces (and hardens) the SnapKV notebook demo
(``notebooks/example.ipynb``): feed a long article as the prefill, ask a
question at the very end, and check the answer. SnapKV compresses the prompt KV
down to ``max_capacity_prompt`` right after prefill; the observation window (the
trailing question tokens) decides which prompt tokens survive, so a relevant
needle should be retained and the model should still answer correctly.

Two modes:

* ``--mode passkey`` (default, rigorous): inject a unique random passkey
  sentence at a chosen depth into the article, then ask for it. Definite ground
  truth -> directly tests whether the needle token survives compression.
* ``--mode notebook``: the original demo question ("What is the repository of
  SnapKV?") against the SnapKV paper text.

Talks to an SGLang server over the OpenAI-compatible ``/v1/chat/completions``
endpoint so the model's chat template is applied.

Usage::

    # against a SnapKV-on server
    python3 needle_snapkv.py --port 30010 --label snapkv_b256 --depth 0.5
    # against a baseline (no compression) server for comparison
    python3 needle_snapkv.py --port 30000 --label baseline --depth 0.5
"""

import argparse
import json
import os
import random
import re
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ARTICLE = os.path.join(_HERE, "data", "snapkv.txt")


def chat(port, content, max_tokens, model="/data/model/Qwen2.5-0.5B-Instruct"):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read())
    text = out["choices"][0]["message"]["content"]
    usage = out.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def insert_needle(article: str, passkey: str, depth: float) -> str:
    """Insert the passkey sentence at ``depth`` fraction through the article."""
    needle = (
        f" The special magic passkey is {passkey}. "
        f"Remember it, it is very important. "
    )
    lines = article.splitlines()
    pos = max(0, min(len(lines), int(len(lines) * depth)))
    lines.insert(pos, needle)
    return "\n".join(lines)


def truncate_words(article: str, max_words: int) -> str:
    if max_words <= 0:
        return article
    words = article.split()
    return " ".join(words[:max_words])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", default=_DEFAULT_ARTICLE)
    ap.add_argument("--port", type=int, default=30010)
    ap.add_argument("--label", default="")
    ap.add_argument("--mode", choices=["passkey", "notebook"], default="passkey")
    ap.add_argument(
        "--depth",
        type=float,
        default=0.5,
        help="Fraction through the article to place the passkey (0=start,1=end).",
    )
    ap.add_argument(
        "--max-words",
        type=int,
        default=0,
        help="Truncate the article to this many words (0=full).",
    )
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--model", default="/data/model/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    with open(args.article) as f:
        article = f.read().strip()
    article = truncate_words(article, args.max_words)

    if args.mode == "passkey":
        rng = random.Random(args.seed)
        passkey = str(rng.randint(10000, 99999))
        doc = insert_needle(article, passkey, args.depth)
        question = (
            "\n\nBased ONLY on the document above, what is the special magic "
            "passkey? Answer with just the number and nothing else."
        )
        gold = passkey
    else:
        doc = article
        question = "\n\nWhat is the repository of SnapKV? Answer concisely."
        gold = None  # no strict ground truth

    content = doc + question

    t0 = time.time()
    answer, ptok, ctok = chat(args.port, content, args.max_tokens, args.model)
    dt = time.time() - t0

    print(f"=== {args.label or args.mode} (port {args.port}) ===")
    print(f"mode        : {args.mode}")
    print(f"prompt_tok  : {ptok}")
    if args.mode == "passkey":
        print(f"depth       : {args.depth}  passkey(gold): {gold}")
    print(f"answer      : {answer.strip()[:300]!r}")
    print(f"completion  : {ctok} tok in {dt:.1f}s")

    if gold is not None:
        found = bool(re.search(rf"\b{re.escape(gold)}\b", answer))
        print(f"RETRIEVED   : {'YES ✅' if found else 'NO ❌'}")
        return 0 if found else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
