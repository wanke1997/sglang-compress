# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import torch
import triton
from tqdm import tqdm

from sglang.srt.layers.quantization.fp8_kernel import (
    _w8a8_block_fp8_matmul,
    _w8a8_block_fp8_matmul_unrolledx4,
)
from sglang.srt.utils import get_device_core_count, get_device_name, is_hip

is_hip_ = is_hip()

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
}


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    config: Dict[str, Any],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise quantization.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.

    Returns:
        torch.Tensor: The result of matmul.
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    # Use manually unrolledx4 kernel on AMD GPU when the grid size is small.
    # Empirical testing shows the sweet spot lies when it's less than the # of
    # compute units available on the device.
    num_workgroups = triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(
        N, config["BLOCK_SIZE_N"]
    )
    kernel = (
        _w8a8_block_fp8_matmul_unrolledx4
        if (is_hip_ == True and num_workgroups <= get_device_core_count())
        else _w8a8_block_fp8_matmul
    )

    kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


def get_rocm_configs_compute_bound():
    configs = []
    waves_per_eu_range = 0
    for num_stages in [2]:
        for block_m in [32, 64, 128, 256]:
            for block_k in [32, 64, 128, 256]:
                for block_n in [16, 32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [1, 4, 8, 16, 32]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu_range,
                                }
                            )
    return configs


def get_configs_compute_bound():
    configs = []
    if is_hip_:
        configs = get_rocm_configs_compute_bound()
    else:
        for num_stages in [2, 3, 4, 5]:
            for block_m in [16, 32, 64, 128, 256]:
                for block_k in [64, 128]:
                    for block_n in [32, 64, 128, 256]:
                        for num_warps in [4, 8]:
                            for group_size in [1, 16, 32, 64]:
                                configs.append(
                                    {
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                    }
                                )
    return configs


def get_weight_shapes(tp_size):
    # NOTE(HandH1998): The weight shapes only works for DeepSeek-V3. Modify them, if you tune for another different model.
    # cannot TP
    total = [
        (512 + 64, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (7168, 16384),
        (7168, 18432),
    ]
    # N can TP
    n_tp = [
        (18432 * 2, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (24576, 1536),
        (4096, 7168),
    ]
    # K can TP
    k_tp = [(7168, 18432), (7168, 16384), (7168, 2048)]

    weight_shapes = []
    for t in total:
        weight_shapes.append(t)
    for n_t in n_tp:
        new_t = (n_t[0] // tp_size, n_t[1])
        weight_shapes.append(new_t)
    for k_t in k_tp:
        new_t = (k_t[0], k_t[1] // tp_size)
        weight_shapes.append(new_t)
    return weight_shapes


def benchmark_config(
    A_fp8, B_fp8, As, Bs, block_size, config, out_dtype=torch.float16, num_iters=10
):
    def run():
        w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, config, out_dtype)

    torch.cuda.synchronize()
    # JIT complication & warmup
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def tune(M, N, K, block_size, out_dtype, search_space):
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(
        torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn
    )

    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(
        torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn
    )

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * factor_for_scale
    Bs = (
        torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda")
        * factor_for_scale
    )

    best_config = None
    best_time = float("inf")
    for config in tqdm(search_space):
        try:
            kernel_time = benchmark_config(
                A_fp8,
                B_fp8,
                As,
                Bs,
                block_size,
                config,
                out_dtype,
                num_iters=10,
            )
        except triton.runtime.autotuner.OutOfResources:
            # Some configurations may be invalid and fail to compile.
            continue

        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    now = datetime.now()
    print(f"{now.ctime()}] Completed tuning for batch_size={M}")
    assert best_config is not None
    return best_config


def save_configs(
    N,
    K,
    block_n,
    block_k,
    configs,
    save_path,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    device_name = get_device_name().replace(" ", "_")
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n}, {block_k}].json"

    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing best config to {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def main(args):
    print(args)

    block_n = args.block_n
    block_k = args.block_k

    tp_size = args.tp_size
    assert args.out_dtype in ["float32", "float16", "bfloat16", "half"]
    out_dtype = DTYPE_MAP[args.out_dtype]
    save_path = args.save_path

    search_space = get_configs_compute_bound()
    search_space = [
        config for config in search_space if block_k % config["BLOCK_SIZE_K"] == 0
    ]

    if args.batch_size is None:
        batch_sizes = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
        ]
    else:
        batch_sizes = [args.batch_size]

    print(f"Start tuning over {len(search_space)} configurations...")

    weight_shapes = get_weight_shapes(tp_size)
    start = time.time()
    for shape in tqdm(weight_shapes):
        N, K = shape[0], shape[1]
        print(f"Tune for weight shape of `N: {N}, K: {K}`")
        benchmark_results = [
            tune(batch_size, N, K, [block_n, block_k], out_dtype, search_space)
            for batch_size in batch_sizes
        ]
        best_configs = {M: config for M, config in zip(batch_sizes, benchmark_results)}
        save_configs(N, K, block_n, block_k, best_configs, save_path)

    end = time.time()
    print(f"Tuning took {end - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tp-size", "-tp", type=int, default=8)
    parser.add_argument(
        "--out-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16", "half"],
        default="float16",
    )
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument(
        "--save-path", type=str, default="python/sglang/srt/layers/quantization/configs"
    )
    args = parser.parse_args()

    main(args)
