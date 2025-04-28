import argparse
import asyncio
import json
import queue
import random
import threading
import time
from typing import Optional

import aiohttp
import requests
from tqdm.asyncio import tqdm

from sglang.bench_serving import (
    RequestFuncOutput,
    get_tokenizer,
    remove_prefix,
    sample_random_requests,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to benchmark concurrent requests to a server."
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=200,
        help="Number of concurrent clients",
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=512,
        help="Length of each new request",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=64,
        help="Length of each output",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of rounds per client",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="poisson",
        choices=["poisson", "uniform"],
        help="Distribution type for request intervals (poisson or uniform)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="Average number of requests per second",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname or IP (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port (default: 30000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="model path compatible with Hugging Face Transformers",
    )
    return parser.parse_args()


async def async_request_sglang_generate(
    payload,
    url,
    pbar: Optional[tqdm] = None,
):
    """
    Sends a streaming request to the server. Gathers text token-by-token.
    """
    async with aiohttp.ClientSession() as session:
        headers = {}
        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()

        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            if data["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text = data["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
            print(f"Request failed: {e}")

    if pbar:
        pbar.update(1)
    return output


def gen_payload(prompt, output_len):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "lora_path": "",
        "return_logprob": False,
        "logprob_start_len": -1,
    }
    return payload


class ReadyQueue:
    """
    Thread-safe queue that can pop requests in different orders based on given policy.
    """

    def __init__(self, init_requests=None, policy="random"):
        self.lock = threading.Lock()
        self.requests = init_requests or []
        self.policy = policy

    def append(self, item):
        with self.lock:
            self.requests.append(item)

    def pop(self):
        with self.lock:
            if not self.requests:
                return None
            if self.policy == "random":
                index = random.randrange(len(self.requests))
                return self.requests.pop(index)
            elif self.policy == "fifo":
                return self.requests.pop(0)
            else:
                # todo, varying thinking time of clients
                raise ValueError(f"{self.policy} not implemented")


class WorkloadGenerator:
    def __init__(self, args):
        # Construct the base URL for requests
        self.url = f"http://{args.host}:{args.port}/generate"

        self.tokenizer = get_tokenizer(args.model)
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.start_time = None
        self.finished_time = None

        self.candidate_inputs = sample_random_requests(
            input_len=args.request_length,
            output_len=args.output_length,
            num_prompts=args.num_clients * args.num_rounds,
            range_ratio=1.0,
            tokenizer=self.tokenizer,
            dataset_path="",
        )
        self.candidate_inputs = [i[0] for i in self.candidate_inputs]

        init_requests = [
            (i, gen_payload(self.candidate_inputs[i], args.output_length))
            for i in range(args.num_clients)
        ]
        self.client_records = {
            i: {"round": 0, "history": init_requests[i][1]["text"]}
            for i in range(args.num_clients)
        }
        self.ready_queue = ReadyQueue(init_requests=init_requests)
        self.candidate_inputs = self.candidate_inputs[args.num_clients :]

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=args.num_clients * args.num_rounds)
        self.performance_metrics = {"ttft": [], "latency": []}

    async def handle_request(self, item):
        try:
            client_id, payload = item
            response = await async_request_sglang_generate(payload, self.url, self.pbar)
            if self.pbar.n == self.pbar.total:
                self.finished_time = time.time()
            self.response_queue.put((client_id, response))
        except Exception as e:
            print(f"Request failed: {e}")

    def request_sender(self):
        async def request_loop():
            while True:
                # Calculate Poisson-distributed wait time
                if self.distribution == "poisson":
                    sleep_time = random.expovariate(self.request_rate)
                elif self.distribution == "uniform":
                    avg_interval = (
                        1.0 / self.request_rate if self.request_rate > 0 else 1.0
                    )
                    sleep_time = random.uniform(0, 2 * avg_interval)
                else:
                    raise ValueError("Invalid distribution type")
                await asyncio.sleep(sleep_time)  # Wait before sending the next request

                new_request = self.ready_queue.pop()
                # Submit async request
                if new_request:
                    asyncio.create_task(self.handle_request(new_request))
                else:
                    if self.pbar.n == self.pbar.total:
                        break

        # Create and run the event loop for asynchronous requests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        loop.close()

    def response_handler(self):
        while True:
            try:
                client_id, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    raise ValueError(f"Request failed with error: {response.error}")
                self.client_records[client_id]["history"] += response.generated_text
                self.client_records[client_id]["round"] += 1
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["latency"].append(response.latency)

                if self.client_records[client_id]["round"] < args.num_rounds:
                    self.client_records[client_id][
                        "history"
                    ] += self.candidate_inputs.pop()
                    self.ready_queue.append(
                        (
                            client_id,
                            gen_payload(
                                self.client_records[client_id]["history"],
                                args.output_length,
                            ),
                        )
                    )
            except queue.Empty:
                if self.pbar.n == self.pbar.total:
                    break

    def run(self):
        request_thread = threading.Thread(target=self.request_sender, daemon=True)
        response_thread = threading.Thread(target=self.response_handler, daemon=True)

        self.start_time = time.time()
        request_thread.start()
        response_thread.start()

        request_thread.join()
        response_thread.join()

        self.pbar.close()
        print("All requests completed.")
        print("Performance metrics summary:")
        print(
            f"  Total requests: {len(self.performance_metrics['ttft'])} at {self.request_rate} requests per second"
        )
        print(
            f"  Average TTFT: {sum(self.performance_metrics['ttft']) / len(self.performance_metrics['ttft']):.2f}"
        )
        print(
            f"  Median TTFT: {sorted(self.performance_metrics['ttft'])[len(self.performance_metrics['ttft']) // 2]:.2f}"
        )
        print(
            f"  Average latency: {sum(self.performance_metrics['latency']) / len(self.performance_metrics['latency']):.2f}"
        )
        print(
            f"  Median latency: {sorted(self.performance_metrics['latency'])[len(self.performance_metrics['latency']) // 2]:.2f}"
        )
        throughput = self.pbar.total / (self.finished_time - self.start_time)
        print(f"Throughput: {throughput:.2f} requests per second")


if __name__ == "__main__":
    args = parse_args()
    flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    for request_rate in range(1, 41, 2):
        args.request_rate = request_rate
        requests.post(flush_cache_url)
        WorkloadGenerator(args).run()
