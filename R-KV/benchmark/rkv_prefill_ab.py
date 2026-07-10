"""R-KV prefill A/B diff-test on REAL captured attention.

Captures per-layer, post-rotary query/key states from a real HF model on a real
long prompt, then compares two R-KV **prefill** compression strategies:

* **Route A (one-shot oracle)** — score the whole prompt once against the true
  final observation window (``rkv.prefill.RKVPrefill.oneshot_keep``).
* **Route B (buffered)** — chunked prefill with buffer-bounded compaction
  (``rkv.prefill.RKVPrefill.buffered_keep``).

The point is to quantify **premature eviction**: tokens the oracle keeps but the
buffered route drops because it committed an eviction decision mid-prefill,
against an observation window that had not yet seen the rest of the prompt.

Metrics reported:

* index overlap (Jaccard, recall of B vs A, #premature evictions);
* **retained attention mass** of the true final observation window over the kept
  KV — i.e. of all the attention the final window pays to the prompt, what
  fraction lands on tokens each route kept. This is the quality-relevant signal
  (A is high by construction; the A-minus-B gap is the premature-eviction cost).

Capture works by wrapping ``apply_rotary_pos_emb`` in the loaded model's
modeling module, so it is model-agnostic (Llama / Mistral / Qwen2 / Qwen3) and
needs no serving stack. Run on one GPU::

    python R-KV/benchmark/rkv_prefill_ab.py \
        --model /data/model/Mistral-7B-Instruct-v0.2 \
        --data /home/sigma/github/kwa-microsoft/benchmark/data/stage1_dev.jsonl \
        --task-type summarization --longest 1 \
        --budget 1024 --window 32 --buffers 128,512,2048 --chunk 4096
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_RKV_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "python",
        "sglang",
        "srt",
        "mem_cache",
        "rkv",
    )
)
sys.path.insert(0, _RKV_DIR)
import prefill as _prefill  # noqa: E402  (loaded by path, GPU-free module)

RKVPrefill = _prefill.RKVPrefill


# --------------------------------------------------------------------------- #
# Capture                                                                      #
# --------------------------------------------------------------------------- #
class RotaryCapture:
    """Wrap ``apply_rotary_pos_emb`` to record per-layer post-rotary q/k."""

    def __init__(self, model) -> None:
        self.module = importlib.import_module(type(model).__module__)
        self._orig = self.module.apply_rotary_pos_emb
        self.q: list[torch.Tensor] = []
        self.k: list[torch.Tensor] = []

    def __enter__(self):
        def wrap(q, k, cos, sin, *args, **kwargs):
            q2, k2 = self._orig(q, k, cos, sin, *args, **kwargs)
            # (bsz, heads, seq, head_dim); batch size 1 for this harness.
            self.q.append(q2[0].detach())
            self.k.append(k2[0].detach())
            return q2, k2

        self.module.apply_rotary_pos_emb = wrap
        return self

    def __exit__(self, *exc):
        self.module.apply_rotary_pos_emb = self._orig
        return False


def load_prompt(path, task_type, longest_rank, tokenizer):
    recs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if task_type and r.get("task_type") != task_type:
            continue
        recs.append(r)
    if not recs:
        raise SystemExit(f"no records with task_type={task_type!r}")

    def ntok(r):
        ids = tokenizer.apply_chat_template(
            r["messages"], add_generation_prompt=True, tokenize=True
        )
        return len(ids)

    recs.sort(key=ntok, reverse=True)
    rec = recs[min(longest_rank, len(recs) - 1)]
    out = tokenizer.apply_chat_template(
        rec["messages"], add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    # Depending on the transformers version this is a tensor or a BatchEncoding.
    if hasattr(out, "shape"):
        ids = out
    elif isinstance(out, dict) or hasattr(out, "get"):
        ids = out["input_ids"]
    else:
        ids = torch.tensor(out).unsqueeze(0)
    return rec, ids


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def diff_metrics(kept_a, kept_b, n):
    a, b = set(kept_a.tolist()), set(kept_b.tolist())
    inter, union = len(a & b), len(a | b)
    return {
        "n": n,
        "budget": len(a),
        "jaccard": inter / union if union else 1.0,
        "recall_b_vs_a": inter / len(a) if a else 1.0,
        "premature_evictions": len(a - b),
    }


@torch.no_grad()
def retained_attention_mass(keys, queries, window_size, kept):
    """Fraction of the final window's attention mass that lands on ``kept``.

    Averaged over layers, kv-groups (q heads), and the ``window_size`` final
    queries. Causal: query at local position j (within the last window) attends
    to keys ``[0, n - window + j]``. Returns a python float in ``[0, 1]``.
    """
    device = keys[0].device
    n = keys[0].shape[-2]
    kept_mask_full = torch.zeros(n, dtype=torch.bool, device=device)
    kept_mask_full[kept] = True

    total = 0.0
    count = 0
    for k_l, q_l in zip(keys, queries):
        kv_heads, _, d = k_l.shape
        q_heads = q_l.shape[0]
        group = q_heads // kv_heads
        wq = q_l[:, -window_size:, :]  # (q_heads, w, d)
        k_exp = k_l.repeat_interleave(group, dim=0)  # (q_heads, n, d)
        logits = torch.matmul(wq, k_exp.transpose(1, 2)) / (d**0.5)  # (q_heads, w, n)
        for j in range(window_size):
            qpos = n - window_size + j
            row = logits[:, j, : qpos + 1]  # (q_heads, qpos+1) causal
            w = torch.softmax(row.float(), dim=-1)
            mass = w[:, kept_mask_full[: qpos + 1]].sum(dim=-1)  # (q_heads,)
            total += mass.mean().item()
            count += 1
    return total / count if count else 1.0


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--task-type", default="summarization")
    ap.add_argument(
        "--longest", type=int, default=0, help="0=longest prompt, 1=2nd, ..."
    )
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--mix-lambda", type=float, default=0.1)
    ap.add_argument("--retain-ratio", type=float, default=0.1)
    ap.add_argument("--buffers", default="128,512,2048")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="sdpa", help="eager is O(n^2) mem; sdpa is safe")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    rec, ids = load_prompt(args.data, args.task_type, args.longest, tok)
    ids = ids.to(args.device)
    n = ids.shape[1]
    print(
        f"model={os.path.basename(args.model)} task={args.task_type} "
        f"prompt_tokens={n} incident={rec.get('incident_id')}"
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation=args.attn
        )
        .to(args.device)
        .eval()
    )

    with RotaryCapture(model) as cap:
        with torch.no_grad():
            model(ids, use_cache=False)
    keys = cap.k  # list[L] of (kv_heads, n, d)
    queries = cap.q  # list[L] of (q_heads, n, d)
    print(
        f"captured {len(keys)} layers | keys {tuple(keys[0].shape)} "
        f"queries {tuple(queries[0].shape)}"
    )
    del model
    torch.cuda.empty_cache()

    pf = RKVPrefill(
        budget=args.budget,
        window_size=args.window,
        kernel_size=args.kernel,
        mix_lambda=args.mix_lambda,
        retain_ratio=args.retain_ratio,
    )
    window_q = [q[:, -args.window :, :] for q in queries]

    print("\n=== Route A (one-shot oracle) ===")
    kept_a = pf.oneshot_keep(keys, window_q)
    mass_a = retained_attention_mass(keys, queries, args.window, kept_a)
    print(f"A: kept={kept_a.numel()} retained_attn_mass={mass_a:.4f}")

    mass_full = retained_attention_mass(
        keys, queries, args.window, torch.arange(n, device=keys[0].device)
    )
    print(f"full-KV retained_attn_mass={mass_full:.4f} (sanity, should be ~1.0)")

    print("\n=== Route B (buffered) sweep vs A ===")
    header = (
        f"{'buffer':>8} {'peak_phys':>10} {'jaccard':>8} {'recall':>8} "
        f"{'premature':>10} {'mass_B':>8} {'mass_gap(A-B)':>14}"
    )
    print(header)
    for buf in [int(x) for x in args.buffers.split(",")]:
        kept_b = pf.buffered_keep(keys, queries, chunk_size=args.chunk, buffer=buf)
        m = diff_metrics(kept_a, kept_b, n)
        mass_b = retained_attention_mass(keys, queries, args.window, kept_b)
        peak = args.budget + buf + args.chunk
        print(
            f"{buf:>8} {peak:>10} {m['jaccard']:>8.4f} "
            f"{m['recall_b_vs_a']:>8.4f} {m['premature_evictions']:>10} "
            f"{mass_b:>8.4f} {mass_a - mass_b:>14.4f}"
        )


if __name__ == "__main__":
    main()
