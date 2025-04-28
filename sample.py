# launch the offline engine

import sys
# sys.path.append('/tmp/kewan/sglang-dev/sglang/python')

import sglang as sgl
import asyncio
import os
import json

if __name__ == "__main__":
    # llm = sgl.Engine(model_path="meta-llama/Llama-3.1-8B")
    compress_algorithm = "CustomKV" # "SnapKV", "CustomKV"
    compress_max_window = 32
    compress_max_prompt = 512
    compress_divide_length = 64
    compress_divide_method = "newline" # "step_length", "newline"
    max_new_tokens = 1024

    # compress_algorithm = "SnapKV" # "SnapKV", "CustomKV"
    # compress_max_window = 32
    # compress_max_prompt = 512
    # compress_divide_length = 64
    # compress_divide_method = "newline" # "step_length", "newline"
    # max_new_tokens = 256
    
    # llm = sgl.Engine(
    #     model_path="meta-llama/Llama-3.1-8B", 
    #     device=device, 
    #     base_gpu_id=base_gpu_id, 
    #     compress_algorithm=compress_algorithm, 
    #     compress_max_window=compress_max_window,
    #     compress_max_prompt=compress_max_prompt,
    # )
    
    # mistralai/Mixtral-8x7B-v0.1
    
    llm = sgl.Engine(
        # model_path="mistralai/Mistral-7B-Instruct-v0.2",
        model_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        compress_algorithm=compress_algorithm, 
        compress_max_window=compress_max_window,
        compress_max_prompt=compress_max_prompt,
        compress_divide_length=compress_divide_length,
        compress_divide_method=compress_divide_method,
        dtype="bfloat16",
    )
    
    with open('snapkv.txt', 'r') as f:
        content = f.read().strip()
    question = "\n What is the repository of SnapKV? [/INST]"

    prompts1 = [
        "How are you today? ",
        "How are you today? ",
        "How are you today? ",
        "How are you today? ",
        # content + question, 
    ]

    sampling_params = {"temperature": 0.85, "top_p": 0.95, "max_new_tokens": max_new_tokens}

    outputs = llm.generate(prompts1, sampling_params)
    for prompt, output in zip(prompts1, outputs):
        print("===============================")
        print(f"Generated text: {output['text']}")


    # file_path = "musique.jsonl"
    # prompts = []

    # with open(file_path, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         # Each line is a JSON object
    #         json_obj = json.loads(line)
    #         context = json_obj["context"]
    #         question = json_obj["input"]
    #         print(len(context), len(question))
    #         prompts.append(context + "\n" + question)
    
    # sampling_params = {"temperature": 0.85, "top_p": 0.95}
    # with open("output.txt", "w") as f:
    #     outputs = llm.generate(prompts, sampling_params)
    #     for prompt, output in zip(prompts, outputs):
    #         f.write("=" * 30 + "\n")
    #         f.write(f"Generated text: {output['text']}")
    #         # print("===============================")
    #         # print(f"Generated text: {output['text']}")
    # print("done")    
    
    # prompts1 = [
    #     "What is the weather like today in LA? Seems like it rained all day long ", # 20
    #     "My name is Ke Wan, and what about you? ", # 14
    #     "How are you today? ", # 7
    # ]
    
    # prompts1 = "Hello, my name is"

    
    # class SamplingParams:
    # def __init__(
    #     self,
    #     max_new_tokens: int = 128,
    #     stop: Optional[Union[str, List[str]]] = None,
    #     stop_token_ids: Optional[List[int]] = None,
    #     temperature: float = 1.0,
    #     top_p: float = 1.0,
    #     top_k: int = -1,
    #     min_p: float = 0.0,
    #     frequency_penalty: float = 0.0,
    #     presence_penalty: float = 0.0,
    #     repetition_penalty: float = 1.0,
    #     min_new_tokens: int = 0,
    #     spaces_between_special_tokens: bool = True,
    #     regex: Optional[str] = None,
    #     n: int = 1,
    #     json_schema: Optional[str] = None,
    #     no_stop_trim: bool = False,
    #     ignore_eos: bool = False,
    #     skip_special_tokens: bool = True,
    # ) -> None:

    # sampling_params = {"temperature": 0.8, "top_p": 0.95}
    # sampling_params = {"temperature": 0.85, "top_p": 0.95}