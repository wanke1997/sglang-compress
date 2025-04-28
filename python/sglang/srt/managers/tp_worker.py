# Copyright 2023-2024 SGLang Team
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
"""A tensor parallel worker."""

import logging
import threading
from typing import Optional, List

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed
from sglang.compress.pyramidkv_utils import SnapKVCluster
from sglang.compress.r1kv_utils import KVCluster

logger = logging.getLogger(__name__)


class TpModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
    ):
        # Parse args
        self.tp_rank = tp_rank

        # Init model and tokenizer
        self.model_config = ModelConfig(
            (
                server_args.model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
        )
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
        )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
                // (server_args.dp_size if server_args.enable_dp_attention else 1)
            ),
            self.model_runner.req_to_token_pool.size,
        )
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)
        if self.tokenizer:
            # For newline compression: get all token ids for "\n"
            self.newline_token_id = self.get_newline_token_ids()
            # For all think token ids: do not compress
            self.think_token_id = self.get_think_token_ids()
        else:
            self.newline_token_id = []
            self.think_token_id = []

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            global_server_args_dict,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_tp_cpu_group(self):
        return self.model_runner.tp_group.cpu_group

    def get_attention_tp_cpu_group(self):
        return self.model_runner.attention_tp_group.cpu_group

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool,
        )

    def get_newline_token_ids(self) -> List[int]:
        newline_token_ids = [
            self.tokenizer.encode("\n")[-1],
            self.tokenizer.encode(".\n")[-1],
            self.tokenizer.encode(")\n")[-1],
            self.tokenizer.encode("\n\n")[-1],
            self.tokenizer.encode(".\n\n")[-1],
            self.tokenizer.encode(")\n\n")[-1],
        ]
        return newline_token_ids
    
    def get_think_token_ids(self) -> List[int]:
        think_token_ids = [
            self.tokenizer.encode("</think>")[-1],
        ]
        return think_token_ids

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.tree_cache = model_worker_batch.tree_cache
        if self.model_runner.server_args.compress_algorithm is not None:
            forward_batch.compress_algorithm = self.model_runner.server_args.compress_algorithm
            forward_batch.compress_max_prompt = self.model_runner.server_args.compress_max_prompt
            forward_batch.compress_max_window = self.model_runner.server_args.compress_max_window
            forward_batch.compress_divide_length = self.model_runner.server_args.compress_divide_length
            forward_batch.compress_divide_method = self.model_runner.server_args.compress_divide_method
            forward_batch.compress_out_locs = {}
            if forward_batch.compress_algorithm == "SnapKV":
                forward_batch.compress_cluster = SnapKVCluster(window_size=forward_batch.compress_max_window, max_capacity_prompt=forward_batch.compress_max_prompt)
            elif forward_batch.compress_algorithm == "CustomKV":
                forward_batch.compress_cluster = SnapKVCluster(window_size=forward_batch.compress_max_window, max_capacity_prompt=forward_batch.compress_max_prompt)
            else:
                raise ValueError(f"Invalid compress algorithm: {forward_batch.compress_algorithm}")
        logits_output = self.model_runner.forward(forward_batch)
        if launch_done:
            launch_done.set()

        if skip_sample:
            next_token_ids = None
            # for newline divide method
            if model_worker_batch.forward_mode.is_decode():
                for idx, req_indice in enumerate(model_worker_batch.req_pool_indices):
                    req_indice = req_indice.item()
                    self.model_runner.req_to_token_pool.newline_compress[req_indice] = False
                    self.model_runner.req_to_token_pool.think_forbid[req_indice] = False
        else:
            next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
            # for newline divide method
            if model_worker_batch.forward_mode.is_decode():
                for idx, req_indice in enumerate(model_worker_batch.req_pool_indices):
                    req_indice = req_indice.item()
                    next_token_id = next_token_ids[idx].item()
                    if next_token_id in self.newline_token_id:
                        self.model_runner.req_to_token_pool.newline_compress[req_indice] = True
                    else:
                        self.model_runner.req_to_token_pool.newline_compress[req_indice] = False
                    if next_token_id in self.think_token_id:
                        self.model_runner.req_to_token_pool.think_forbid[req_indice] = True
                    else:
                        self.model_runner.req_to_token_pool.think_forbid[req_indice] = False
        return logits_output, next_token_ids

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        embeddings = logits_output.embeddings
        return embeddings

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path, recv_req.load_format
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.name, recv_req.dtype, recv_req.shape
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = self.model_runner.update_weights_from_tensor(
            MultiprocessingSerializer.deserialize(recv_req.serialized_named_tensors)
        )
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter
