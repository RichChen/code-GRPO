"""OpenAI-compatible chat client used as an episode policy (a local vLLM server or a paid API)."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from arl.env.episode import Message, PolicyContextExceeded

TOOL_CALL_MARKER = re.compile(r"<tool_call>|\"name\"\s*:\s*\"(read_file|search_code|list_dir|edit_file|run_tests)\"")
DOTENV = Path(".env")


def read_dotenv(path: str | Path = DOTENV) -> dict[str, str]:
    """``KEY=VALUE`` lines of a .env file; blank lines and ``#`` comments are skipped."""
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def env_value(key: str, dotenv: str | Path = DOTENV) -> str | None:
    """An existing environment variable wins over the .env file."""
    return os.environ.get(key) or read_dotenv(dotenv).get(key) or None


@dataclass
class ClientConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    api_key: str = "EMPTY"
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 1024
    timeout_s: float = 300.0
    retries: int = 5
    # read {env_prefix}_BASE_URL / _MODEL / _API_KEY from the environment or .env
    env_prefix: str | None = None
    extra_body: dict[str, Any] | None = None
    # let the HTTP client pick up HTTPS_PROXY etc. from the environment
    trust_env: bool = False
    # CNY per million tokens: {"cache_hit": ..., "cache_miss": ..., "output": ...}
    price_cny_per_mtok: dict[str, float] | None = None
    # "chat": /v1/chat/completions (OpenAIPolicy); "tokens": /v1/completions with token ids in and out
    # (arl.rollout.tokenclient.TokenPolicy), prompts rendered locally with this tokenizer
    endpoint: str = "chat"
    tokenizer: str = "Qwen/Qwen3-1.7B"
    # None: the server's default (the model's generation_config); 0 disables top-k
    top_k: int | None = None
    # tokens endpoint only: compare every rendered prompt with the server's /tokenize
    verify_tokenize: bool = False

    @classmethod
    def load(cls, path: str | Path | None = None, dotenv: str | Path = DOTENV, **overrides: Any) -> "ClientConfig":
        data: dict[str, Any] = {}
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown client config keys: {sorted(unknown)}")
        prefix = data.get("env_prefix")
        if prefix:
            for name in ("base_url", "model", "api_key"):
                key = f"{prefix}_{name.upper()}"
                value = env_value(key, dotenv)
                if not value:
                    raise ValueError(f"{key} is not set in the environment or {dotenv}")
                data[name] = value
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("api_key")
        return d

    def cost_cny(self, usage: "Usage") -> float:
        p = self.price_cny_per_mtok
        if not p:
            return 0.0
        miss = usage.prompt_tokens - usage.cache_hit_tokens
        return (usage.cache_hit_tokens * p["cache_hit"] + miss * p["cache_miss"]
                + usage.completion_tokens * p["output"]) / 1e6


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0  # summed over calls
    completion_tokens: int = 0  # summed over calls
    cache_hit_tokens: int = 0  # summed over calls; DeepSeek's prompt_cache_hit_tokens, else 0
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    finish_reasons: dict[str, int] = field(default_factory=dict)
    unparsed_tool_calls: int = 0  # replies that look like tool calls the server did not parse

    @property
    def final_context_tokens(self) -> int:
        return self.last_prompt_tokens + self.last_completion_tokens


def _is_context_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "context length" in text or "maximum context" in text or "too long" in text or "max_model_len" in text


class OpenAIPolicy:
    """``policy(messages, tools) -> assistant message`` backed by a chat completions endpoint.

    One instance per episode, so ``usage`` describes that episode. A reply whose text looks like a
    tool call the server failed to parse is returned as a malformed tool call (no function name),
    so the episode records a format error and continues instead of treating it as a submission.
    """

    def __init__(self, config: ClientConfig, client: Any | None = None, seed: int | None = None):
        import openai  # not installed in the training venv, which only uses the token-level client

        self.openai = openai
        self.config = config
        self.client = client or openai.OpenAI(
            base_url=config.base_url, api_key=config.api_key, timeout=config.timeout_s, max_retries=0,
            http_client=openai.DefaultHttpxClient(trust_env=config.trust_env))
        self.seed = seed
        self.usage = Usage()

    def _create(self, messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.config.model, messages=messages,
            temperature=self.config.temperature, top_p=self.config.top_p, max_tokens=self.config.max_tokens,
        )
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        if self.seed is not None:
            kwargs["seed"] = self.seed + self.usage.calls
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body
        delay = 2.0
        for attempt in range(self.config.retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except self.openai.BadRequestError as exc:
                if _is_context_error(exc):
                    raise PolicyContextExceeded(str(exc)) from exc
                raise
            except (self.openai.APIConnectionError, self.openai.APITimeoutError, self.openai.InternalServerError,
                    self.openai.RateLimitError):
                if attempt == self.config.retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise AssertionError("unreachable")

    def __call__(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        resp = self._create(messages, tools)
        choice = resp.choices[0]
        u = resp.usage
        self.usage.calls += 1
        if u is not None:
            self.usage.prompt_tokens += u.prompt_tokens
            self.usage.completion_tokens += u.completion_tokens
            self.usage.last_prompt_tokens = u.prompt_tokens
            self.usage.last_completion_tokens = u.completion_tokens
            self.usage.cache_hit_tokens += getattr(u, "prompt_cache_hit_tokens", None) or 0
        reason = choice.finish_reason or "none"
        self.usage.finish_reasons[reason] = self.usage.finish_reasons.get(reason, 0) + 1

        msg = choice.message
        out: Message = {"role": "assistant", "content": msg.content}
        calls = []
        for call in msg.tool_calls or []:
            calls.append({"id": call.id, "type": "function",
                          "function": {"name": call.function.name, "arguments": call.function.arguments}})
        if not calls and msg.content and TOOL_CALL_MARKER.search(msg.content):
            self.usage.unparsed_tool_calls += 1
            calls.append({"id": f"unparsed_{self.usage.calls}", "type": "function",
                          "function": {"name": "", "arguments": msg.content[:2000]}})
        if calls:
            out["tool_calls"] = calls
        return out
