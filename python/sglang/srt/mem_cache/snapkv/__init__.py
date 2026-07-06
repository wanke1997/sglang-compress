"""SnapKV prompt-phase KV-cache compression for SGLang v0.5.14.

* :mod:`.algo` — the pure, device-agnostic SnapKV algorithm (CPU-testable,
  bit-comparable to the reference ``SnapKVCluster``).
* :mod:`.integration` — the SGLang runtime bridge (prefill hook, paged-pool
  physical eviction, logical rotary positions).
"""

from sglang.srt.mem_cache.snapkv.algo import SnapKVCluster  # noqa: F401
