"""``arl.rollout.hermes.parse`` against vLLM 0.31's own hermes parser (run in the vLLM venv)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("vllm")

from arl.rollout.hermes import parse  # noqa: E402
from arl.sft.data import Renderer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CALL = '<tool_call>\n{"name": "read_file", "arguments": {"path": "pkg/a.py"}}\n</tool_call>'
EDGE = [
    "", " ", "Fixed it.", CALL, "I'll read it.\n" + CALL, " \n" + CALL, CALL + "\ntrailing",
    CALL + "\n" + '<tool_call>\n{"name": "run_tests", "arguments": {}}',
    '<tool_call>{"name":"search_code","arguments":{"pattern":"é\\n"}}</tool_call>',
    "<tool_call>\nnot json\n</tool_call>", '<tool_call>{"arguments": {}}</tool_call>',
    '<tool_call>{"name": "run_tests"}</tool_call>', '<tool_call>["run_tests"]</tool_call>',
    '<tool_call>{"name": 3, "arguments": {}}</tool_call>', "<tool_call></tool_call>", "<tool_call>",
    CALL + "\n<tool_call>{broken", '<tool_call>{"name": "x", "arguments": "{\\"a\\": 1}"}</tool_call>',
    '<tool_call>{"name": "x", "arguments": [1, 2]}</tool_call>', "</tool_call>" + CALL,
]


@pytest.fixture(scope="module")
def renderer():
    return Renderer()


def vllm_parse(renderer, text):
    """vLLM's chat path for tool_choice auto: ``extract_tool_calls`` then the whitespace-content rule."""
    info = renderer.parser.extract_tool_calls(text, None)
    if not info.tools_called:
        return text, []
    content = info.content if info.content and info.content.strip() else None
    return content, [(c.function.name, c.function.arguments) for c in info.tool_calls]


def corpus(renderer):
    texts = list(EDGE)
    path = ROOT / "data/sft/v2/train.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            s = json.loads(line)
            texts.append(renderer.tok.decode(s["input_ids"][s["loss_start"]:-1]))
    return texts


def test_parse_matches_vllm_hermes_parser(renderer):
    texts = corpus(renderer)
    diffs = [t for t in texts if parse(t) != vllm_parse(renderer, t)]
    assert not diffs, (len(diffs), diffs[:3])
