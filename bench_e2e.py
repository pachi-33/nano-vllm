"""End-to-end benchmark for nano-vllm, focused on edge/local deployment scenarios.

Modes:
  latency  — TTFT, prefill/decode speed, E2E latency per scenario
  memory   — model weights, activation, KV cache, peak VRAM breakdown
  stress   — sustained decode for N minutes, track throttling
  quality  — perplexity on wikitext2 + sanity check

Workloads:
  short_chat  — 50-100 token prompt, 200 token output
  doc_qa      — 2048-4096 token prompt, 100 token output
  code_gen    — 200 token prompt, 1024 token output
"""
import argparse
import json
import os
import statistics
import sys
from time import perf_counter

import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


# ── Workload definitions ──────────────────────────────────────────

WORKLOADS = {
    "short_chat": {
        "prompt_len": 80,
        "max_tokens": 200,
        "num_prompts": 50,
        "description": "Short in, short out — cold-start and absolute latency",
    },
    "doc_qa": {
        "prompt_len": 3000,
        "max_tokens": 100,
        "num_prompts": 10,
        "description": "Long in, short out — prefill speed and KV cache memory",
    },
    "code_gen": {
        "prompt_len": 200,
        "max_tokens": 1024,
        "num_prompts": 5,
        "description": "Medium in, long out — decode stability and memory leaks",
    },
}

SANITY_CHECKS = [
    ("What is 2+2?", "4"),
    ("What is the capital of France?", "Paris"),
    ("What color is the sky on a clear day?", "blue"),
    ("How many days are in a week?", "7"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("How many legs does a spider have?", "8"),
    ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
    ("What is the square root of 144?", "12"),
    ("Who painted the Mona Lisa?", "da Vinci"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many continents are there?", "7"),
    ("What is the fastest land animal?", "cheetah"),
    ("What is H2O commonly known as?", "water"),
    ("Who discovered gravity?", "Newton"),
    ("What is the capital of Japan?", "Tokyo"),
    ("How many sides does a hexagon have?", "6"),
    ("What animal is known as the King of the Jungle?", "lion"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("Who invented the telephone?", "Bell"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the chemical symbol for oxygen?", "O"),
    ("How many hours are in a day?", "24"),
    ("What is the capital of China?", "Beijing"),
    ("What is the smallest prime number?", "2"),
    ("What season comes after winter?", "spring"),
    ("Who wrote 'The Cat in the Hat'?", "Seuss"),
    ("What is the hardest natural substance?", "diamond"),
]


def make_prompt_token_ids(prompt_len: int, num_prompts: int) -> list[list[int]]:
    """Generate random token IDs for benchmarking."""
    import random
    random.seed(42)
    vocab_size = 151936  # Qwen3 vocab
    return [[random.randint(0, vocab_size - 1) for _ in range(prompt_len)] for _ in range(num_prompts)]


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


# ── Latency mode ──────────────────────────────────────────────────

def run_latency(llm, workload_name: str) -> dict:
    from nanovllm import SamplingParams
    wl = WORKLOADS[workload_name]
    prompt_ids_list = make_prompt_token_ids(wl["prompt_len"], wl["num_prompts"])
    sp = SamplingParams(temperature=0.6, max_tokens=wl["max_tokens"])

    # Warmup
    llm.generate([prompt_ids_list[0]], sp, use_tqdm=False)

    ttfts = []
    e2es = []
    prefill_speeds = []
    decode_speeds = []

    # Submit all requests at once, step through to capture per-step metrics
    t_submit = perf_counter()
    for prompt_ids in prompt_ids_list:
        llm.add_request(prompt_ids, sp)

    while not llm.is_finished():
        t = perf_counter()
        output, num_tokens = llm.step()
        elapsed = perf_counter() - t

        if num_tokens > 0:
            ttfts.append(elapsed)
            prefill_speeds.append(num_tokens / elapsed)
        elif num_tokens < 0:
            decode_speeds.append(-num_tokens / elapsed)

        for seq_id, token_ids in output:
            e2es.append(perf_counter() - t_submit)

    result = {
        "workload": workload_name,
        "num_prompts": wl["num_prompts"],
        "prompt_len": wl["prompt_len"],
        "max_tokens": wl["max_tokens"],
        "ttft_ms": {
            "avg": statistics.mean(ttfts) * 1000 if ttfts else 0,
            "p50": pct(ttfts, 50) * 1000 if ttfts else 0,
            "p99": pct(ttfts, 99) * 1000 if ttfts else 0,
        },
        "prefill_tok_s": statistics.mean(prefill_speeds) if prefill_speeds else 0,
        "decode_tok_s": {
            "avg": statistics.mean(decode_speeds) if decode_speeds else 0,
            "p50": pct(decode_speeds, 50) if decode_speeds else 0,
        },
        "e2e_s": {
            "avg": statistics.mean(e2es) if e2es else 0,
            "p50": pct(e2es, 50) if e2es else 0,
        },
    }
    return result


def print_latency(result: dict):
    print(f"\n[Latency] Scenario: {result['workload']} ({result['num_prompts']} prompts)")
    print(f"  Prompt len:    {result['prompt_len']} tokens")
    print(f"  Max output:    {result['max_tokens']} tokens")
    ttft = result["ttft_ms"]
    print(f"  TTFT:          avg={ttft['avg']:.1f}ms  p50={ttft['p50']:.1f}ms  p99={ttft['p99']:.1f}ms")
    print(f"  Prefill:       {result['prefill_tok_s']:.0f} tok/s")
    dec = result["decode_tok_s"]
    print(f"  Decode:        avg={dec['avg']:.0f} tok/s  p50={dec['p50']:.0f} tok/s")
    e2e = result["e2e_s"]
    print(f"  E2E latency:   avg={e2e['avg']:.2f}s  p50={e2e['p50']:.2f}s")


# ── Memory mode ───────────────────────────────────────────────────

def run_memory(llm, workload_name: str) -> dict:
    from nanovllm import SamplingParams
    wl = WORKLOADS[workload_name]
    prompt_ids_list = make_prompt_token_ids(wl["prompt_len"], 1)
    sp = SamplingParams(temperature=0.6, max_tokens=wl["max_tokens"])

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    llm.generate(prompt_ids_list, sp, use_tqdm=False)
    torch.cuda.synchronize()

    stats = torch.cuda.memory_stats()
    total_vram = torch.cuda.get_device_properties(0).total_memory

    result = {
        "workload": workload_name,
        "model_weights_gb": stats.get("allocated_bytes.all.peak", 0) / 2**30 * 0.3,  # rough estimate
        "peak_allocated_gb": stats["allocated_bytes.all.peak"] / 2**30,
        "current_allocated_gb": stats["allocated_bytes.all.current"] / 2**30,
        "peak_total_gb": torch.cuda.max_memory_allocated() / 2**30,
        "total_vram_gb": total_vram / 2**30,
    }
    return result


def print_memory(result: dict):
    print(f"\n[Memory] Scenario: {result['workload']}")
    print(f"  Peak allocated:    {result['peak_allocated_gb']:.2f} GB")
    print(f"  Current allocated: {result['current_allocated_gb']:.2f} GB")
    print(f"  Peak total VRAM:   {result['peak_total_gb']:.2f} GB")
    print(f"  Total VRAM:        {result['total_vram_gb']:.2f} GB")


# ── Stress mode ───────────────────────────────────────────────────

def run_stress(llm, duration_s: int) -> dict:
    from nanovllm import SamplingParams
    prompt_ids = make_prompt_token_ids(100, 1)[0]
    sp = SamplingParams(temperature=0.6, max_tokens=256, ignore_eos=True)

    # Warmup
    llm.generate([prompt_ids], sp, use_tqdm=False)

    samples = []
    start = perf_counter()
    iteration = 0
    while perf_counter() - start < duration_s:
        t = perf_counter()
        llm.generate([prompt_ids], sp, use_tqdm=False)
        elapsed = perf_counter() - t
        tokens = sp.max_tokens
        decode_speed = tokens / elapsed
        elapsed_total = perf_counter() - start
        samples.append({"time_s": elapsed_total, "speed_tok_s": decode_speed})
        iteration += 1

    # Sample per minute
    minute_samples = {}
    for s in samples:
        minute = int(s["time_s"] / 60) + 1
        minute_samples.setdefault(minute, []).append(s["speed_tok_s"])

    minute_avgs = {}
    for m in sorted(minute_samples):
        minute_avgs[m] = statistics.mean(minute_samples[m])

    first_avg = minute_avgs.get(1, 0)
    last_avg = minute_avgs.get(max(minute_avgs), 0)
    throttle = (last_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0

    result = {
        "duration_s": duration_s,
        "iterations": iteration,
        "minute_avgs": minute_avgs,
        "throttle_pct": throttle,
    }
    return result


def print_stress(result: dict):
    print(f"\n[Stress] Duration: {result['duration_s']}s ({result['iterations']} iterations)")
    for m, avg in result["minute_avgs"].items():
        print(f"  Minute {m:2d}:  {avg:.0f} tok/s")
    print(f"  Throttle:   {result['throttle_pct']:+.1f}%")


# ── Quality mode ──────────────────────────────────────────────────

def run_perplexity(model_path: str) -> float | None:
    """Compute perplexity on Wikitext-2 using a simple sliding window."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        print("  [skip] Install `datasets` and `transformers` for PPL testing")
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    token_ids = tokenizer.encode(text)
    print(f"  Wikitext-2 test: {len(token_ids)} tokens")

    # Use PyTorch directly for PPL (model forward, not nano-vllm engine)
    from transformers import AutoModelForCausalLM
    import torch.nn.functional as F

    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
    model.eval()

    window = 2048
    stride = 512
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(token_ids) - window, stride):
            chunk = token_ids[i:i + window]
            input_ids = torch.tensor([chunk], device=device)
            logits = model(input_ids).logits
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
            if i % (stride * 10) == 0:
                print(f"    ... {i}/{len(token_ids)} tokens processed")

    ppl = float(torch.exp(torch.tensor(total_loss / total_tokens)))
    del model
    torch.cuda.empty_cache()
    return ppl


def run_sanity_check(llm) -> dict:
    from nanovllm import SamplingParams
    sp = SamplingParams(temperature=0.6, max_tokens=64)
    passed = 0
    failures = []

    for prompt, keyword in SANITY_CHECKS:
        outputs = llm.generate([prompt], sp, use_tqdm=False)
        output_text = outputs[0]["text"].lower()
        if keyword.lower() in output_text:
            passed += 1
        else:
            failures.append((prompt, keyword, output_text[:100]))

    result = {
        "total": len(SANITY_CHECKS),
        "passed": passed,
        "failures": failures,
    }
    return result


def print_quality(ppl: float | None, sanity: dict):
    print(f"\n[Quality]")
    if ppl is not None:
        print(f"  PPL (wikitext2):  {ppl:.2f}")
    print(f"  Sanity Check:     {sanity['passed']}/{sanity['total']} passed")
    if sanity["failures"]:
        print(f"  Failures:")
        for prompt, keyword, output in sanity["failures"][:5]:
            print(f"    Q: {prompt[:50]}  expected: {keyword}  got: {output[:60]}...")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="nano-vllm edge deployment benchmark")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B_fp16/"))
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "latency", "memory", "stress", "quality"])
    parser.add_argument("--workload", type=str, default="all",
                        choices=["all"] + list(WORKLOADS.keys()))
    parser.add_argument("--duration", type=int, default=900, help="Stress test duration in seconds")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    results = {}

    gpu_name = torch.cuda.get_device_properties(0).name
    print(f"=== nano-vllm Benchmark Report ===")
    print(f"Model: {os.path.basename(args.model)} | GPU: {gpu_name} | PP: {args.pipeline_parallel_size} | TP: {args.tensor_parallel_size}")
    print(f"Mode: {args.mode} | Workload: {args.workload}")

    modes = ["latency", "memory", "stress", "quality"] if args.mode == "all" else [args.mode]
    workloads = list(WORKLOADS.keys()) if args.workload == "all" else [args.workload]

    # Create a single LLM instance shared across all modes to avoid
    # "trying to initialize the default process group twice!" — PyTorch
    # does not allow dist.init_process_group() after destroy_process_group().
    llm = LLM(args.model, enforce_eager=args.enforce_eager,
               pipeline_parallel_size=args.pipeline_parallel_size,
               tensor_parallel_size=args.tensor_parallel_size)

    for mode in modes:
        if mode == "latency":
            results["latency"] = {}
            for wl in workloads:
                r = run_latency(llm, wl)
                print_latency(r)
                results["latency"][wl] = r

        elif mode == "memory":
            results["memory"] = {}
            for wl in workloads:
                r = run_memory(llm, wl)
                print_memory(r)
                results["memory"][wl] = r

        elif mode == "stress":
            r = run_stress(llm, args.duration)
            print_stress(r)
            results["stress"] = r

        elif mode == "quality":
            # PPL uses HF model directly
            ppl = run_perplexity(args.model) if args.pipeline_parallel_size == 1 else None
            sanity = run_sanity_check(llm)
            print_quality(ppl, sanity)
            results["quality"] = {"ppl": ppl, "sanity": {"passed": sanity["passed"], "total": sanity["total"]}}

    del llm
    torch.cuda.empty_cache()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
