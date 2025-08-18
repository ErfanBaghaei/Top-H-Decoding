import argparse
import json
from tqdm import tqdm
import datasets
import torch
from transformers import pipeline
from logit_processor import TopH_LogitsProcessor
from itertools import islice



def generate_responses(
    data,
    model_name: str,
    max_new_tokens: int = 1024,
    temperature: float = 2.0,
    top_p: float | None = None,
    do_sample: bool = True,
    item_cap: int | None = None,
    alpha: float = 0.4, 
):
    print("Generating responses...")
    responses = []
    inputs_ = []

    # build generation kwargs, skipping None so HF doesn't complain
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
    }
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="balanced",
    )

    logit_processor = TopH_LogitsProcessor(top_n = 100, temperature = gen_kwargs["temperature"], alpha = alpha)


    iterable = islice(data, item_cap) if item_cap is not None else data
    try:
        total = item_cap if item_cap is not None else len(data)
    except TypeError:
        total = item_cap  # fall back if data has no __len__

    for idx, item in enumerate(tqdm(iterable, total=total), start=1):
        input_text = item["instruction"]
        inputs_.append(input_text)

        messages = [{"role": "user", "content": input_text}]
        result = pipe(messages, **gen_kwargs, logits_processor = [logit_processor])

        assistant_response = None
        try:
            # Case A: text-generation style -> list[{"generated_text": str}]
            if isinstance(result, list) and isinstance(result[0].get("generated_text"), str):
                assistant_response = result[0]["generated_text"]

            # Case B: chat-style -> list[{"generated_text": [{"role": "...", "content": "..."}]}]
            elif isinstance(result, list) and isinstance(result[0].get("generated_text"), list):
                for out in result[0]["generated_text"]:
                    if isinstance(out, dict) and out.get("role") == "assistant":
                        assistant_response = out.get("content")
                        break
        except Exception:
            assistant_response = None

        response = assistant_response if assistant_response is not None else ""
        responses.append(response)
        
        if item_cap is not None and idx >= item_cap:
            break

    return responses, inputs_


def save_responses(output_file, responses, inputs, model_name: str):
    print(f"Saving responses to {output_file}...")
    if len(responses) != len(inputs):
        raise ValueError("Responses and inputs lists must have the same length.")

    data = []
    for response, input_ in zip(responses, inputs):
        data.append(
            {
                "output": response,
                "instruction": input_,
                "generator": model_name,  # reflect the actual model used
                "dataset": "oasst",
                "datasplit": "eval",
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def evaluate_models(
    model_name: str,
    save_address: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    do_sample: bool,
    item_cap: int | None,
    alpha: float,
):
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    responses, inputs = generate_responses(
        eval_set,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        item_cap=item_cap,
    )
    save_responses(save_address, responses, inputs, model_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model and save generations.")
    parser.add_argument(
        "--save_address",
        required=True,
        help="Where to write the output JSON (replaces output_file_1).",
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id to load (replaces model_1_name).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p nucleus sampling (0–1). Omit to disable.",
    )
    parser.add_argument(
        "--do_sample",
        dest="do_sample",
        action="store_true",
        help="Enable sampling (default).",
    )
    parser.add_argument(
        "--no_do_sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling; use greedy/beam search.",
    )
    parser.add_argument(
        "--item_cap",
        type=int,
        default=None,
        help="Max number of items to process (e.g., 100). Omit for no cap.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="alpha",
    )

    parser.set_defaults(do_sample=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_models(
        model_name=args.model_name,
        save_address=args.save_address,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        item_cap=args.item_cap,
        alpha = args.alpha,
    )
