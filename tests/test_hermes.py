import json

import pytest

from arl.rollout.hermes import parse
from arl.sft.data import template_arguments

CALL = '<tool_call>\n{"name": "read_file", "arguments": {"path": "pkg/a.py"}}\n</tool_call>'


def test_plain_text_is_content_without_tool_calls():
    assert parse("Fixed it.") == ("Fixed it.", [])
    assert parse("") == ("", [])


def test_calls_with_and_without_leading_content():
    assert parse(CALL) == (None, [("read_file", '{"path": "pkg/a.py"}')])
    assert parse("I'll read it.\n" + CALL) == ("I'll read it.\n", [("read_file", '{"path": "pkg/a.py"}')])
    assert parse(" \n" + CALL)[0] is None


def test_several_calls_and_an_unclosed_last_call():
    text = CALL + "\n" + '<tool_call>\n{"name": "run_tests", "arguments": {}}'
    assert parse(text) == (None, [("read_file", '{"path": "pkg/a.py"}'), ("run_tests", "{}")])


def test_arguments_are_reserialised_without_ascii_escapes():
    text = '<tool_call>{"name":"search_code","arguments":{"pattern":"é"}}</tool_call>'
    assert parse(text) == (None, [("search_code", '{"pattern": "é"}')])


@pytest.mark.parametrize("bad", [
    "<tool_call>\nnot json\n</tool_call>",
    '<tool_call>{"arguments": {}}</tool_call>',
    '<tool_call>{"name": "run_tests"}</tool_call>',
    '<tool_call>["run_tests"]</tool_call>',
    '<tool_call>{"name": 3, "arguments": {}}</tool_call>',
    "<tool_call></tool_call>",
    CALL + "\n<tool_call>{broken",
])
def test_any_bad_block_means_no_tool_calls_and_the_whole_text_as_content(bad):
    assert parse("x " + bad) == ("x " + bad, [])


def test_template_arguments_coerce_like_vllm():
    assert template_arguments('{"path": "a"}') == {"path": "a"}
    assert template_arguments({"path": "a"}) == {"path": "a"}
    for bad in ("", None, "not json", "[1]", "42", json.dumps("x")):
        assert template_arguments(bad) == {}
