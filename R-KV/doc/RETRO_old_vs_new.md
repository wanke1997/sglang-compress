# R-KV on SGLang — 旧实现复盘（old vs new）

对比对象：

- **旧实现**：历史 commit `c076b4092f072dca1281f7f1b9276905a5f0cd2f`
  （分支 `review/sglang-c076b409`）。把整个 SGLang fork 源码 vendored 进仓库，
  R-KV 改动直接改在 `SGLang/python/sglang/...` 里。
- **新实现**：`main`。干净的集成层 `SGLang/rkv/`（`algo.py` + `integration.py`）
  + `SGLang/patch/rkv-sglang-0.5.14.patch`。

> 说明：旧 commit 与 main 在 `SGLang/` 路径下 diff 高达 ~22101 个文件，
> 主要是因为旧 commit vendored 了整份上游源码；真正的 R-KV 逻辑集中在少数文件。

---

## 1. 旧实现的架构

- **算法**：`SGLang/python/sglang/compress/r1kv_utils.py` 的 `KVCluster.update_kv`
  （HF 风格：per-head 打分 → `topk` → `gather` → 拼接最近窗口）。
- **接入点**：全部塞进 `RadixAttention`
  （`SGLang/python/sglang/srt/layers/radix_attention.py`）：
  - 每一层 forward 里调用 `compress()`：对该层 K/V 跑 `update_kv`，物化出新的压缩 K/V，
    `_get_new_kv_loc` 在 layer 0 分配新 slot、其余层复用，`set_kv_buffer` 写入。
  - 最后一层结束后 `_apply_compress`：`free` 旧 slot、把新 slot 写回 `req_to_token`、
    记录 `tree_cache.compressed_req_len`。
  - 下一次 `prepare_for_decode` 才把 `seq_lens` / `req.seq_len` 改成压缩后的长度。
- **触发**：`need_compress()`。prefill 长度超阈值即触发；decode 用 `"newline"` 或
  `"step_length"`（`forward_batch.steps % compress_divide_length`）。

## 2. 新实现的架构（对照）

- 纯算法 `rkv/algo.py`（新增 `select_indices`）与集成层 `rkv/integration.py` 解耦，可单测/可对拍。
- 跨 head = mean、跨 layer = sum，归约成**单一全局 per-token 驱逐决策**。
- 物理压缩在**整个 forward 之后**原地进行（`maybe_compact`：保留 slot 前移、`free` 尾部）。
- **Scheme A**：`seq_lens` 只跟踪物理 KV 长度；RoPE 位置单独用 `logical_position` /
  `override_decode_positions` 覆盖成逻辑位置。
- 长度对账当场完成：`pending_length_updates` 由 scheduler 在同一个 forward 之后
  `take_pending_length_updates()` 立刻应用；`on_request_end` 清理请求状态。

---

## 3. 致命 bug #1：RoPE 位置塌缩

**现象**：压缩后输出乱码 / 循环。

**机制**：旧实现把 `seq_lens` 同时当作两个语义——(a) 物理 KV 长度、(b) 新 token 的 RoPE 位置来源
（`positions = clamp_position(seq_lens)`）。压缩把 `seq_lens` 从几千砍到 `budget`，于是：

- 保留下来的 key 仍带着**原始绝对位置**烤进去的旋转（RoPE 在写入 KV 前已固化）。
- 下一个 query 的位置却被算成 `≈budget`。
- query 位置 < 最新上下文 key 的位置 → 相对偏移变负 → 注意力错乱，逐步累积。

**背景知识**：RoPE 是**相对**的——注意力打分只依赖 query 位置 `m` 与 key 位置 `n` 的差
`m-n`，前提是二者旋转在同一条位置坐标轴上。key 的位置已固化在存储里无法免费改。

**正确做法（R-KV，中间驱逐）**：只剔除被驱逐 token，**保留剩余 token 的原始 RoPE，不重新分配位置**；
唯一要保证的是 query 的位置继续按**逻辑（未压缩）时间线**递增。→ 新实现的 scheme A。
（重新编号 / position-shifting 需要对所有存活 key 重新施加 RoPE，代价高且改变语义，不适合中间驱逐。）

## 4. 致命 bug #2：token slot 记账不一致（泄露 / 双重释放 / 串号）

**根因**：真正释放/重分配 slot 的时刻（forward 中途 `_apply_compress`）与更新长度记账的时刻
（下一步 `prepare_for_decode`，通过 `compressed_req_len` 延迟对账，且以**可复用的 `req_pool_idx`** 为 key，
请求结束时**不清理**）是脱节的。任何"按长度 free"的路径在这个窗口里都会出错：

1. **`compressed_req_len` 结束时不清理 + key 复用 → 泄露/串号**
   请求 A 在"压缩那一步"结束 → `compressed_req_len[idxA]` 永远留在 map（prepare_for_decode 不再处理已结束的 A）。
   `idxA` 被回收复用给新请求 B → B 首个 decode 命中陈旧记录，把 `seq_lens` 强行改成 `budget`。
   若 B 真实长度 > budget，`req_to_token[budget:真实长度]` 的 slot 被记账丢弃、永不 free → **泄露**；反之读脏 → **损坏**。

2. **finish / retract 撞上压缩步 → 双重释放**
   `cache_finished_req` 用 `token_id_len = req.seq_len`（仍是压缩前的大值 L）释放 `req_to_token[:L]`
   = `[budget 个新 slot]` + `[budget:L 的陈旧旧 slot id]`；后半段已在 `_apply_compress` 里 free 过 → **双重释放**。
   retract（`retract_decode` 用 `seq_lens_cpu[idx]`）同理。双重释放使同一 slot 索引在 free 列表出现两次
   → available_size 虚高 + 两个请求共用同一 slot → KV 串号。

**新实现为什么没有**：原地压缩不 alloc 新块；`_compact_request` 当场更新
`req.kv_committed_len/kv_allocated_len`，并通过 `pending_length_updates` 让 scheduler 在同一 forward 后
**立刻**对账，不存在跨步窗口；`on_request_end` 明确清理状态，不受 `req_pool_idx` 复用影响。

---

## 5. 其余 bug

### A. 影响正确性 / 精度

- **#3 per-head 各选各的驱逐集，与"单一 per-token 位置/槽映射"根本不兼容。**
  `update_kv` 的 `indices` 形状 `(bsz, kv_heads, budget-window)`，每个 head 各自 `gather` 不同原始 token。
  但 `req_to_token` 与 `positions` 每 token 只有一个值，无法同时表示不同 head 的不同原始位置。
  **这是 bug #1 即使想修也修不了的更深一层原因**——必须先跨 head 归约（新实现 `cross-head=mean`）。

- **#4 超参硬编码，用户 flag 不生效且偏离参考值。**
  `tp_worker.py` 每次 forward 新建
  `KVCluster(attn_pooling="max", pooling="maxpool", mix_alpha=0.03, retain_ratio=0.2)`，
  server_args 未暴露这些。参考实现是 `mix_lambda=0.07/0.1, retain_ratio=0.1` → 精度对不上且无法调参。

- **#8 observation window 未填满就打分。**
  `update_kv` 用最后 `window_size` 个 query 当观测者，旧实现无任何"window 已填满"守卫；
  prefill 触发或首次 decode 触发时 query 窗口可能不足/零初始化。
  新实现用 `assert buffer_size >= window_size` 防这个。

### B. 触发逻辑错误

- **#6 decode `step_length` 用 batch 级 `forward_batch.steps`，不是 per-request。**
  `steps` 是整个 batch 共享的计数器，合并 batch 时取 `min`。所有请求在同一全局步长上一起压缩，
  batch 组成变化即抖动/重置。新实现是 per-request 的 `steps_since_compact`。

- **#7 `think_forbid` 逻辑不完整 / 方向反。**
  只在生成 `</think>` 的那一步置 True（下一步复位），仅屏蔽一步压缩；未跟踪"是否处于 think 区间"，
  达不到"思考段不压缩"的意图。

- **#5 budget 与触发阈值被同一参数绑死。**
  `compress_max_prompt` 既是"超过即压缩"的阈值，又是 `max_capacity_prompts`（保留预算），
  无法设成"阈值 ≠ budget"。且 help 文本写 default 2048/64，实际 512/32（文档不一致）。

### C. 健壮性 / 延迟隐患

- **#9 prefill 压缩后 `prefix_indices` / ChunkCache entry 指向已释放的槽。**
  `_apply_compress` 已 free 原始 prompt 槽并缩到 budget，但 `cache_unfinished_req` 仍把
  `req_to_token[:len(fill_ids)]`（完整 prompt 长度）存进 entry 并赋给 `req.prefix_indices` → 指向已释放槽。

- **#10 `newline`/`think` token id 提取脆弱 + `req_pool_idx` 复用带来陈旧标志。**
  `tokenizer.encode("\n")[-1]` 在不同 tokenizer 下可能取错 id；
  `newline_compress`/`think_forbid` 以可复用的 `req_pool_idx` 为 key，新请求可能继承陈旧 True。

- **#11 decode 里对同一槽 `set_kv_buffer` 写两次。**
  `forward()` 先 `set_kv_buffer(cache_loc, k, v)`，随后 `attn_backend.forward(..., save_kv_cache=True)`
  又写一次。数据相同不致错，但反映 KV 写入所有权不清晰。

- **#12 alloc-before-free 峰值。**
  先 `alloc(budget)` 再释放旧的，峰值 = 旧 + 新，靠 `model_runner` "预留一块内存" 兜底，高负载下更易 OOM。

---

## 6. 一句话总结

旧实现真正的两个致命问题是：**(1)** 把物理 KV 长度和 RoPE 位置绑在同一个 `seq_lens` 上，
压缩一发生位置就塌缩；**(2)** slot 的释放/重分配与长度记账在时序上脱节、且记账 key 复用后不清理，
导致泄露 / 双重释放 / 串号。更深一层，**(3)** per-head 各选各的驱逐集，使正确的单一位置根本无法定义。
新实现通过"物理长度 vs 逻辑位置解耦 + 跨 head/layer 归约成单一全局决策 + 原地压缩 + 同步对账 + 生命周期清理"
逐条消除了这些问题。

---

## 7. 复现对比用的命令

```bash
# 从历史 commit fork 出对比分支
git branch review/sglang-c076b409 c076b4092f072dca1281f7f1b9276905a5f0cd2f

# 旧实现的 R-KV 相关文件
git ls-tree -r --name-only review/sglang-c076b409 -- SGLang/python/sglang/compress/
git grep -l -iE "compress|r1kv|rkv" review/sglang-c076b409 -- 'SGLang/python/sglang/srt/**' | grep -v 3rdparty

# 新实现
ls SGLang/rkv/ SGLang/patch/ SGLang/tests/
```
