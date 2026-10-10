"""Token-level client against a fake server; the renderer is a stub, so no tokenizer is needed."""

import json

import pytest

from arl.env.episode import PolicyContextExceeded, run_episode
from arl.env.tools import TOOL_SCHEMAS
from arl.rollout.calibrate import make_policy, token_stats
from arl.rollout.client import ClientConfig, OpenAIPolicy
from arl.rollout.tokenclient import TokenPolicy, server_root


class StubRenderer:
    def render(self, messages, generation_prompt):
        return json.dumps(messages, sort_keys=True) + ("|gen" if generation_prompt else "")

    def encode(self, text):
        return [len(text) % 1000, 7]


class Resp:
    def __init__(self, status, body):
        self.status_code, self.body = status, body
        self.text = json.dumps(body)

    def json(self):
        return self.body


class FakeServer:
    """Replies to /v1/completions with scripted texts; records request bodies."""

    def __init__(self, texts, errors=()):
        self.texts, self.errors, self.bodies, self.paths = list(texts), list(errors), [], []

    def post(self, url, json):
        self.paths.append(url)
        self.bodies.append(json)
        if self.errors:
            return self.errors.pop(0)
        if url.endswith("/tokenize"):
            return Resp(200, {"tokens": [1, 2]})
        text = self.texts.pop(0) if self.texts else "done"
        ids = [100 + i for i in range(len(text) % 5 + 1)] + [151645]
        return Resp(200, {"choices": [{"text": text, "token_ids": ids, "finish_reason": "stop",
                                       "logprobs": {"token_logprobs": [-0.5] * len(ids)}}]})


def policy(texts, **cfg):
    config = ClientConfig(endpoint="tokens", model="m", retries=2, **cfg)
    server = FakeServer(texts)
    return TokenPolicy(config, seed=10, renderer=StubRenderer(), client=server), server


def test_request_carries_token_ids_and_sampling_settings():
    p, server = policy(["hi"], top_k=0, temperature=1.0)
    msg = p([{"role": "user", "content": "x"}], TOOL_SCHEMAS)
    body = server.bodies[0]
    assert server.paths[0] == "http://127.0.0.1:8000/v1/completions"
    assert body["prompt"] == p.turns[0]["prompt_ids"] and body["model"] == "m"
    assert body["logprobs"] == 0 and body["return_token_ids"] is True and body["skip_special_tokens"] is False
    assert body["top_k"] == 0 and body["temperature"] == 1.0 and body["seed"] == 10
    assert msg == {"role": "assistant", "content": "hi"}
    p([{"role": "user", "content": "y"}], TOOL_SCHEMAS)
    assert server.bodies[1]["seed"] == 11
    p, server = policy(["a"])
    p([{"role": "user", "content": "x"}], TOOL_SCHEMAS)
    assert "top_k" not in server.bodies[0]  # the server's default (the model's generation_config)


def test_turn_records_and_usage():
    p, _ = policy(["abc"])
    p([{"role": "user", "content": "x"}], TOOL_SCHEMAS)
    t = p.turns[0]
    assert t["completion_ids"][-1] == 151645 and len(t["completion_logprobs"]) == len(t["completion_ids"])
    assert t["finish_reason"] == "stop"
    assert p.usage.calls == 1 and p.usage.completion_tokens == len(t["completion_ids"])
    assert p.usage.prompt_tokens == len(t["prompt_ids"]) and p.usage.finish_reasons == {"stop": 1}
    stats = token_stats(p)
    assert stats["turn_completion_tokens"] == [len(t["completion_ids"])] and "turn_tokens" not in stats
    assert token_stats(p, with_tokens=True)["turn_tokens"] == p.turns


def test_tool_calls_and_unparsed_calls_become_messages_like_the_chat_client():
    call = '<tool_call>\n{"name": "run_tests", "arguments": {}}\n</tool_call>'
    p, _ = policy(["Run.\n" + call, '{"name": "run_tests" oops'])
    m1 = p([{"role": "user", "content": "x"}], TOOL_SCHEMAS)
    assert m1["content"] == "Run.\n" and m1["tool_calls"][0]["function"] == {"name": "run_tests", "arguments": "{}"}
    m2 = p([{"role": "user", "content": "x"}], TOOL_SCHEMAS)
    assert m2["tool_calls"][0]["function"]["name"] == "" and p.usage.unparsed_tool_calls == 1


def test_context_errors_and_retries(monkeypatch):
    p, server = policy(["ok"])
    server.errors = [Resp(400, {"message": "This model's maximum context length is 8192 tokens"})]
    with pytest.raises(PolicyContextExceeded):
        p([{"role": "user", "content": "x"}], TOOL_SCHEMAS)
    p, server = policy(["ok"])
    server.errors = [Resp(500, {"message": "boom"})]
    p.config.retries = 1
    monkeypatch.setattr("arl.rollout.tokenclient.time.sleep", lambda s: None)
    assert p([{"role": "user", "content": "x"}], TOOL_SCHEMAS)["content"] == "ok"
    p, server = policy(["ok"])
    server.errors = [Resp(404, {"message": "no such model"})]
    with pytest.raises(RuntimeError, match="404"):
        p([{"role": "user", "content": "x"}], TOOL_SCHEMAS)


def test_tokenize_verification_counts_mismatches():
    p, server = policy(["ok"], verify_tokenize=True)
    p([{"role": "user", "content": "x"}], TOOL_SCHEMAS)
    assert server.paths[0] == "http://127.0.0.1:8000/tokenize"
    assert server.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert p.tokenize_checked == 1 and p.tokenize_mismatches == 1  # the stub renders [n, 7], the fake says [1, 2]


def test_thinking_mode_is_refused_and_endpoints_are_dispatched():
    with pytest.raises(ValueError):
        TokenPolicy(ClientConfig(endpoint="tokens", extra_body={"chat_template_kwargs": {"enable_thinking": True}}),
                    renderer=StubRenderer(), client=FakeServer([]))
    assert server_root("http://h:1/v1/") == "http://h:1"
    with pytest.raises(ValueError):
        make_policy(ClientConfig(endpoint="bogus"), None)
    pytest.importorskip("openai")  # the chat client is not installed in the train venv
    assert isinstance(make_policy(ClientConfig(), None), OpenAIPolicy)


def test_runs_a_whole_episode():
    pytest.importorskip("httpx2")  # test_rollout's toy tasks need the main venv
    from test_rollout import toy

    task = toy(0)
    p, server = policy(['<tool_call>\n{"name": "list_dir", "arguments": {"path": "."}}\n</tool_call>', "Done."])
    result = run_episode(task, p)
    assert result.turns == 3 and result.nudged  # "Done." before any edit is nudged once, then "done" submits
    assert [m["role"] for m in result.messages][:4] == ["system", "user", "assistant", "tool"]
    assert len(p.turns) == 3 and all(b["prompt"] == t["prompt_ids"] for b, t in zip(server.bodies, p.turns))
