import asyncio
import json
import time
import uuid
from threading import Thread
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


# ── Pydantic request/response models ──────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 4096
    stream: bool = False
    stop: list[str] | None = None


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str | list[str]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 64
    stream: bool = False
    stop: list[str] | None = None
    n: int = 1


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class DeltaContent(BaseModel):
    content: str | None = None
    role: str | None = None


class StreamChoice(BaseModel):
    index: int
    delta: DeltaContent
    finish_reason: str | None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "nano-vllm"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ── Server app ──────────────────────────────────────────────────────────────

class NanoVLLMServer:

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.app = FastAPI(title="nano-vllm OpenAI-compatible API")
        self._register_routes()

    def _register_routes(self):
        self.app.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"])
        self.app.add_api_route("/v1/completions", self.completions, methods=["POST"])
        self.app.add_api_route("/v1/models", self.list_models, methods=["GET"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/", self.health, methods=["GET"])

    async def health(self):
        return {"status": "ok"}

    async def list_models(self):
        return ModelListResponse(data=[
            ModelInfo(id=self.engine.model_name, created=int(time.time()))
        ])

    async def chat_completions(self, request: ChatCompletionRequest):
        # Build prompt from messages using chat template
        prompt = self.engine.tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        if request.stream:
            return StreamingResponse(
                self._stream_chat(prompt, sampling_params, request_id, created),
                media_type="text/event-stream",
            )

        # Non-streaming: use generate
        outputs = self.engine.generate([prompt], sampling_params, use_tqdm=False)
        text = outputs[0]["text"]
        token_ids = outputs[0]["token_ids"]
        prompt_tokens = len(self.engine.tokenizer.encode(prompt))

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=self.engine.model_name,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(token_ids),
                total_tokens=prompt_tokens + len(token_ids),
            ),
        )

    async def _stream_chat(
        self, prompt: str, sampling_params: SamplingParams, request_id: str, created: int
    ) -> AsyncIterator[str]:
        _ERROR = object()  # sentinel that cannot collide with model output

        # First chunk with role
        chunk = ChatCompletionChunk(
            id=request_id, created=created, model=self.engine.model_name,
            choices=[StreamChoice(index=0, delta=DeltaContent(role="assistant"), finish_reason=None)],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        # Run generation in a thread and collect tokens via queue
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run():
            try:
                for token_id, text, finished in self.engine.generate_stream(prompt, sampling_params):
                    loop.call_soon_threadsafe(queue.put_nowait, (text, finished))
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, (_ERROR, str(e)))

        thread = Thread(target=_run, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            text, finished = item
            if text is _ERROR:
                error_msg = finished  # second element is the error string
                error_chunk = {
                    "id": request_id, "object": "chat.completion.chunk", "created": created,
                    "model": self.engine.model_name,
                    "choices": [{"index": 0, "delta": {"content": f"\n[Error: {error_msg}]"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                break
            chunk = ChatCompletionChunk(
                id=request_id, created=created, model=self.engine.model_name,
                choices=[StreamChoice(index=0, delta=DeltaContent(content=text), finish_reason=None)],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        chunk = ChatCompletionChunk(
            id=request_id, created=created, model=self.engine.model_name,
            choices=[StreamChoice(index=0, delta=DeltaContent(), finish_reason="stop")],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    async def completions(self, request: CompletionRequest):
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )

        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        if request.stream:
            # For completions streaming, stream the first prompt
            return StreamingResponse(
                self._stream_completion(prompts[0], sampling_params, request_id, created),
                media_type="text/event-stream",
            )

        outputs = self.engine.generate(prompts, sampling_params, use_tqdm=False)
        choices = []
        total_prompt = 0
        total_completion = 0
        for i, out in enumerate(outputs):
            prompt_tokens = len(self.engine.tokenizer.encode(prompts[i]))
            total_prompt += prompt_tokens
            total_completion += len(out["token_ids"])
            choices.append({
                "index": i,
                "text": out["text"],
                "finish_reason": "stop",
            })

        return {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": self.engine.model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            },
        }

    async def _stream_completion(
        self, prompt: str, sampling_params: SamplingParams, request_id: str, created: int
    ) -> AsyncIterator[str]:
        _ERROR = object()

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run():
            try:
                for token_id, text, finished in self.engine.generate_stream(prompt, sampling_params):
                    loop.call_soon_threadsafe(queue.put_nowait, (text, finished))
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, (_ERROR, str(e)))

        thread = Thread(target=_run, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            text, finished = item
            if text is _ERROR:
                break
            chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": self.engine.model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        chunk = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": self.engine.model_name,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
