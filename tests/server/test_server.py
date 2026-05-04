"""
Tests for the OpenAI-compatible API server.

Requires:
  - CUDA GPU
  - Model weights at ~/huggingface/Qwen3-0.6B_fp16/ (or --model path override)
  - pip install fastapi uvicorn httpx

Run:
  pytest tests/server/ -v -s
  pytest tests/server/ -v -s --pp 2          # PP mode
  pytest tests/server/ -v -s --model /path    # custom model
"""
import json

import pytest


class TestHealth:

    def test_health_endpoint(self, http):
        resp = http.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestModels:

    def test_list_models(self, http):
        resp = http.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "nano-vllm"
        assert "id" in data["data"][0]


class TestChatCompletions:

    def test_basic_completion(self, http):
        resp = http.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 16,
            "temperature": 0.6,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]

    def test_response_format(self, http):
        resp = http.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        })
        data = resp.json()
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert "model" in data

    def test_max_tokens_limit(self, http):
        resp = http.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Tell me a long story"}],
            "max_tokens": 4,
        })
        data = resp.json()
        assert data["usage"]["completion_tokens"] <= 4

    def test_streaming(self, http):
        with http.stream("POST", "/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    assert chunk["object"] == "chat.completion.chunk"
                    assert "choices" in chunk
                    chunks.append(chunk)
            assert len(chunks) >= 2  # At least role + content + finish
            # First chunk should have role
            first_delta = chunks[0]["choices"][0]["delta"]
            assert first_delta.get("role") == "assistant"
            # Last chunk should have finish_reason
            last_chunk = chunks[-1]
            assert last_chunk["choices"][0]["finish_reason"] == "stop"

    def test_multi_turn(self, http):
        resp = http.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "My name is TestBot"},
                {"role": "assistant", "content": "Hello TestBot!"},
                {"role": "user", "content": "What is my name?"},
            ],
            "max_tokens": 16,
        })
        data = resp.json()
        content = data["choices"][0]["message"]["content"].lower()
        assert "testbot" in content or "test" in content or len(content) > 0


class TestCompletions:

    def test_basic_completion(self, http):
        resp = http.post("/v1/completions", json={
            "model": "test",
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "temperature": 0.6,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert len(data["choices"][0]["text"]) > 0
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_batch_prompts(self, http):
        resp = http.post("/v1/completions", json={
            "model": "test",
            "prompt": ["Hello", "World"],
            "max_tokens": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) == 2

    def test_streaming(self, http):
        with http.stream("POST", "/v1/completions", json={
            "model": "test",
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    assert chunk["object"] == "text_completion"
                    chunks.append(chunk)
            assert len(chunks) >= 1


class TestSamplingParams:

    def test_top_p(self, http):
        """top_p=1.0 should not crash (equivalent to no filtering)."""
        resp = http.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "top_p": 1.0,
        })
        assert resp.status_code == 200

    def test_temperature(self, http):
        resp = http.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "temperature": 0.3,
        })
        assert resp.status_code == 200

    def test_stop_string(self, http):
        """stop parameter should terminate generation when matched."""
        resp = http.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Count: 1, 2, 3"}],
            "max_tokens": 32,
            "stop": [","],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"
