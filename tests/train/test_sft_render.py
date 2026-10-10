"""Rendering checks against the real Qwen3 tokenizer and vLLM's hermes parser.

Run in the vLLM venv: ``PYTHONPATH=src HF_HUB_OFFLINE=1 ~/.venvs/vllm/bin/python -m pytest tests/train``.
Skipped when transformers, vLLM or the cached tokenizer is missing (e.g. in the main project venv).
"""

import json

import pytest

pytest.importorskip("transformers")
pytest.importorskip("vllm.tool_parsers.hermes_tool_parser")

from arl.env.episode import initial_messages  # noqa: E402
from arl.sft.data import MODEL, Renderer, template_messages  # noqa: E402
from conftest import make_task  # noqa: E402


@pytest.fixture(scope="module")
def renderer():
    try:
        return Renderer(MODEL)
    except OSError as exc:
        pytest.skip(f"tokenizer for {MODEL} not available: {exc}")


def call(i, name, args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def trajectory(first_content=""):
    msgs = initial_messages(make_task())
    msgs += [
        {"role": "assistant", "content": first_content, "tool_calls": [call(0, "search_code", {"pattern": "def add"})]},
        {"role": "tool", "tool_call_id": "c0", "name": "search_code", "content": "pkg/nums.py:1: def add(a, b):"},
        {"role": "assistant", "content": "", "tool_calls": [
            call(1, "read_file", {"path": "pkg/nums.py"}), call(2, "list_dir", {"path": "tests"})]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": " 1| def add(a, b):\n 2|     return a - b"},
        {"role": "tool", "tool_call_id": "c2", "name": "list_dir", "content": "test_visible.py"},
        {"role": "assistant", "content": None, "tool_calls": [
            call(3, "edit_file", {"path": "pkg/nums.py", "old_string": "a - b", "new_string": "a + b  # ✓"})]},
        {"role": "tool", "tool_call_id": "c3", "name": "edit_file", "content": "Edited pkg/nums.py"},
        {"role": "assistant", "content": "Fixed `add`."},
    ]
    return {"task_id": "toy", "k": 0, "messages": msgs}


def test_every_turn_is_consistent_and_masks_only_the_prefix(renderer):
    samples, failure = renderer.turn_samples(trajectory("I'll search first."))
    assert failure is None and len(samples) == 4
    tok = renderer.tok
    for s in samples:
        prefix = tok.decode(s["input_ids"][: s["loss_start"]])
        target = tok.decode(s["input_ids"][s["loss_start"]:])
        assert prefix.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert target.endswith("<|im_end|>") and "<think>" not in target
        assert s["input_ids"][-1] == renderer.im_end
    assert tok.decode(samples[1]["input_ids"][samples[1]["loss_start"]:]).count("<tool_call>") == 2
    # earlier turns appear in later prefixes without the empty think block, as the template renders history
    later_prefix = tok.decode(samples[3]["input_ids"][: samples[3]["loss_start"]])
    assert later_prefix.count("<think>") == 1 and "a + b  # ✓" in later_prefix


def test_whole_trajectory_as_one_sequence_would_not_match(renderer):
    msgs = template_messages(trajectory()["messages"])
    whole = renderer.render(msgs, False)
    first = next(j for j, m in enumerate(msgs) if m["role"] == "assistant")
    assert not whole.startswith(renderer.render(msgs[:first], True))


def test_leading_newline_content_is_reported_inconsistent(renderer):
    samples, failure = renderer.turn_samples(trajectory("\nLooking."))
    assert samples == [] and failure == "template_text"


def test_hermes_parse_mismatch_is_detected(renderer):
    m = template_messages(trajectory()["messages"])[2]
    assert renderer.parse_matches(
        '<tool_call>\n{"name": "search_code", "arguments": {"pattern": "def add"}}\n</tool_call>', m)
    assert not renderer.parse_matches(
        '<tool_call>\n{"name": "search_code", "arguments": {"pattern": "def sub"}}\n</tool_call>', m)
