import argparse
import os

# Disable torch.compile to avoid Triton compatibility issues
# (e.g. Triton 2.x + PyTorch 2.11). Must be set before importing torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import uvicorn

from nanovllm.llm import LLM
from nanovllm.server import NanoVLLMServer


def main():
    parser = argparse.ArgumentParser(description="nano-vllm OpenAI-compatible API server")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="Tensor parallel size (default: 1)")
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1, help="Pipeline parallel size (default: 1)")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph capture")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization (default: 0.9)")
    args = parser.parse_args()

    engine = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    server = NanoVLLMServer(engine)
    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
