"""Token-level rollout client (audit revision 2A-2, item 3): the one client for calibration, GRPO and evaluation.

Each turn:

1. the conversation is rendered with ``arl.sft.data.Renderer`` (Qwen3 chat template, the tool schemas,
   ``enable_thinking=False``, generation prompt), the renderer the SFT data was verified with;
2. the token ids go to vLLM's ``/v1/completions`` as ``prompt``, with ``return_token_ids`` and ``logprobs=0``
   (vLLM 0.31: ``choices[0].token_ids`` and ``choices[0].logprobs.token_logprobs``); ``skip_special_tokens`` is
   off, as vLLM's hermes parser sets it for a chat request with tools;
3. the text is parsed by ``arl.rollout.hermes``, which reproduces the chat endpoint's tool-call handling.

The policy keeps, per turn, the prompt ids, the sampled ids, their logprobs and the finish reason: training
uses exactly these tokens and never renders the conversation again. Like ``OpenAIPolicy`` it is a callable
``policy(messages, tools) -> assistant message`` with a ``usage``, so ``run_episode`` and the calibration code
take it unchanged. Needs transformers (the tokenizer) and httpx, not openai: it runs in the vLLM venv and in
the training venv.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from arl.env.episode import Message, PolicyContextExceeded
from arl.env.tools import TOOL_SCHEMAS
from arl.rollout import hermes
from arl.rollout.client import TOOL_CALL_MARKER, ClientConfig, Usage

try:
    import httpx
except ImportError:  # the main project venv only has openai's httpx2
    import httpx2 as httpx  # type: ignore[no-redef]

_renderers: dict[str, Any] = {}
_clients: dict[tuple[float, bool], Any] = {}
_shared_lock = threading.Lock()
_render_lock = threading.Lock()


def shared_renderer(tokenizer: str) -> Any:
    from arl.sft.data import Renderer

    with _shared_lock:
        if tokenizer not in _renderers:
            _renderers[tokenizer] = Renderer(tokenizer)
        return _renderers[tokenizer]


def shared_client(timeout_s: float, trust_env: bool) -> Any:
    with _shared_lock:
        key = (timeout_s, trust_env)
        if key not in _clients:
            _clients[key] = httpx.Client(timeout=timeout_s, trust_env=trust_env,
                                         limits=httpx.Limits(max_connections=64, max_keepalive_connections=64))
        return _clients[key]


def server_root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def is_context_error(text: str) -> bool:
    text = text.lower()
    return "context length" in text or "maximum context" in text or "too long" in text or "max_model_len" in text


class TokenPolicy:
    """``policy(messages, tools) -> assistant message`` backed by ``/v1/completions`` with token ids.

    One instance per episode. ``turns`` holds one record per model call: ``prompt_ids``, ``completion_ids``,
    ``completion_logprobs``, ``finish_reason``. With ``config.verify_tokenize`` every prompt is also compared
    with the server's ``/tokenize`` of the same conversation (``tokenize_checked`` / ``tokenize_mismatches``).
    """

    def __init__(self, config: ClientConfig, seed: int | None = None, renderer: Any = None, client: Any = None,
                 model: str | None = None):
        kwargs = (config.extra_body or {}).get("chat_template_kwargs") or {}
        if kwargs.get("enable_thinking", False):
            raise ValueError("the token-level client renders prompts with enable_thinking=False")
        self.config = config
        self.model = model or config.model
        self.seed = seed
        self.renderer = renderer or shared_renderer(config.tokenizer)
        self.http = client or shared_client(config.timeout_s, config.trust_env)
        self.root = server_root(config.base_url)
        self.usage = Usage()
        self.turns: list[dict[str, Any]] = []
        self.tokenize_checked = 0
        self.tokenize_mismatches = 0

    def prompt_ids(self, messages: list[Message]) -> list[int]:
        from arl.sft.data import template_messages

        with _render_lock:  # one tokenizer is shared by the episode threads
            return self.renderer.encode(self.renderer.render(template_messages(messages), True))

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        delay = 2.0
        for attempt in range(self.config.retries + 1):
            try:
                resp = self.http.post(f"{self.root}{path}", json=body)
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == self.config.retries:
                    raise
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 400 and is_context_error(resp.text):
                    raise PolicyContextExceeded(resp.text[:500])
                if resp.status_code < 500 and resp.status_code != 429 or attempt == self.config.retries:
                    raise RuntimeError(f"{path} returned {resp.status_code}: {resp.text[:500]}")
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise AssertionError("unreachable")

    def _check_tokenize(self, messages: list[Message], tools: list[dict[str, Any]], ids: list[int]) -> None:
        body = {"model": self.model, "messages": messages, "tools": tools, "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        self.tokenize_checked += 1
        if self._post("/tokenize", body)["tokens"] != ids:
            self.tokenize_mismatches += 1

    def __call__(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        if tools != TOOL_SCHEMAS:
            raise ValueError("prompts are rendered with arl.env.tools.TOOL_SCHEMAS")
        ids = self.prompt_ids(messages)
        if self.config.verify_tokenize:
            self._check_tokenize(messages, tools, ids)
        body: dict[str, Any] = {
            "model": self.model, "prompt": ids, "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature, "top_p": self.config.top_p,
            "logprobs": 0, "return_token_ids": True, "skip_special_tokens": False,
        }
        if self.config.top_k is not None:
            body["top_k"] = self.config.top_k
        if self.seed is not None:
            body["seed"] = self.seed + self.usage.calls
        choice = self._post("/v1/completions", body)["choices"][0]
        completion = list(choice["token_ids"])
        logprobs = list(choice["logprobs"]["token_logprobs"])
        if len(logprobs) != len(completion):
            raise RuntimeError(f"{len(logprobs)} logprobs for {len(completion)} tokens")
        reason = choice.get("finish_reason") or "none"
        self.turns.append({"prompt_ids": ids, "completion_ids": completion, "completion_logprobs": logprobs,
                           "finish_reason": reason})
        u = self.usage
        u.calls += 1
        u.prompt_tokens += len(ids)
        u.completion_tokens += len(completion)
        u.last_prompt_tokens, u.last_completion_tokens = len(ids), len(completion)
        u.finish_reasons[reason] = u.finish_reasons.get(reason, 0) + 1
        return self.message(choice["text"])

    def message(self, text: str) -> Message:
        """The assistant message the chat endpoint would have returned for this text (see ``OpenAIPolicy``)."""
        content, calls = hermes.parse(text)
        out: Message = {"role": "assistant", "content": content}
        tool_calls = [{"id": f"call_{self.usage.calls}_{k}", "type": "function",
                       "function": {"name": name, "arguments": args}} for k, (name, args) in enumerate(calls)]
        if not tool_calls and content and TOOL_CALL_MARKER.search(content):
            self.usage.unparsed_tool_calls += 1
            tool_calls.append({"id": f"unparsed_{self.usage.calls}", "type": "function",
                               "function": {"name": "", "arguments": content[:2000]}})
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out
