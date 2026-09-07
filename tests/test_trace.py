from pathlib import Path
from types import SimpleNamespace

from root.agent import AgentResult, Step, ToolCall
from root.trace import (
    StreamGate,
    format_answer,
    render_step,
    render_summary,
    render_turn,
)


def tool_step(result="395", seconds=0.4, forced=False):
    return Step(
        index=0,
        generation="<call>",
        seconds=seconds,
        prompt_tokens=120,
        generated_tokens=18,
        call=ToolCall("calculator", "'17 * 23 + 4'", result),
        tool_seconds=0.02,
        forced=forced,
    )


def answer_step():
    return Step(
        index=1,
        generation="The result is 395.",
        seconds=0.3,
        prompt_tokens=140,
        generated_tokens=9,
    )


def test_trace_off_prints_nothing():
    assert render_step(tool_step(), "off") == []


def test_tool_step_shows_call_result_and_cost():
    lines = render_step(tool_step(), "on")
    assert lines[0] == "  ● calculator('17 * 23 + 4')"
    assert lines[1] == "    ⎿ 395"
    assert "0.4s · 18 tok" in lines[2]


def test_answer_step_is_quiet_until_full():
    assert render_step(answer_step(), "on") == []
    assert "answer" in render_step(answer_step(), "full")[0]


def test_full_level_shows_the_raw_generation():
    lines = render_step(answer_step(), "full")
    assert any("raw: The result is 395." in line for line in lines)


def test_forced_call_is_labelled():
    assert "forced" in render_step(tool_step(forced=True), "on")[2]


def test_long_results_are_flattened_and_truncated():
    lines = render_step(tool_step(result="x\n" * 400), "on")
    assert lines[1].endswith("...")
    assert "\n" not in lines[1]


def test_summary_counts_calls_and_flags_a_toolless_agent():
    result = AgentResult(agent="chat", prompt="p", answer="a", steps=[answer_step()])
    summary = render_summary(result, tool_count=0)
    assert "0 calls" in summary
    assert "agent has no tools" in summary


def test_summary_reports_the_step_limit():
    result = AgentResult(
        agent="calc", prompt="p", answer="a", steps=[tool_step()], hit_step_limit=True
    )
    assert "step limit reached" in render_summary(result, tool_count=1)
    assert "1 call" in render_summary(result, tool_count=1)


def test_render_turn_covers_every_step():
    result = AgentResult(agent="calc", prompt="p", answer="a", steps=[tool_step(), answer_step()])
    lines = render_turn(result, "full")
    assert any("calculator" in line for line in lines)
    assert any("answer" in line for line in lines)


def test_color_wraps_the_meta_line_only():
    lines = render_step(tool_step(), "on", color=True)
    assert "\033[2m" in lines[2]
    assert "\033[2m" not in lines[0]


def test_a_fenced_block_is_indented_and_the_fences_removed():
    answer = "Here:\n```python\ndef f():\n    return 1\n```\ndone"
    assert format_answer(answer).splitlines() == [
        "Here:",
        "    python",
        "    def f():",
        "        return 1",
        "done",
    ]


def test_inline_markup_is_stripped_without_colour():
    assert format_answer("It **doubles** the `x`") == "It doubles the x"


def test_inline_markup_becomes_escape_codes_with_colour():
    rendered = format_answer("It **doubles** the `x`", color=True)
    assert "\033[1m" in rendered
    assert "**" not in rendered


def test_plain_prose_is_unchanged():
    assert format_answer("The answer is 42.") == "The answer is 42."


def gate(marker="<|tool_call_start|>"):
    written = []
    return StreamGate(marker, written.append), written


def test_prose_streams_through():
    stream, written = gate()
    for chunk in ["The ", "answer ", "is 42."]:
        stream.feed(chunk)
    stream.close()
    assert "".join(written) == "The answer is 42."


def test_a_tool_call_is_never_shown():
    stream, written = gate()
    for chunk in ["<|tool_call", "_start|>[calculator(", "expression='6 * 7')]"]:
        stream.feed(chunk)
    stream.close()
    assert written == []
    assert stream.suppressed


def test_text_before_a_tool_call_is_still_held_back_mid_marker():
    stream, written = gate()
    stream.feed("Sure. <|tool")
    assert "".join(written) == "Sure. "
    stream.feed("_call_start|>[x()]")
    stream.close()
    assert "".join(written) == "Sure. "


def test_a_partial_marker_that_turns_out_to_be_prose_is_released():
    stream, written = gate()
    stream.feed("a < b")
    stream.close()
    assert "".join(written) == "a < b"


def test_a_format_without_a_marker_streams_everything():
    stream, written = gate(marker="")
    stream.feed("plain text")
    stream.close()
    assert "".join(written) == "plain text"


def test_nothing_is_emitted_twice_on_close():
    stream, written = gate()
    stream.feed("done")
    stream.close()
    stream.close()
    assert "".join(written) == "done"


def test_a_command_that_raises_does_not_end_the_session(monkeypatch, capsys):
    """The loop owns the model and the conversation; a broken command must not
    take them down with it."""
    from root import terminal

    session = SimpleNamespace(
        running=True,
        trace="off",
        stream=False,
        agent="chat",
        model=SimpleNamespace(model_id="a/b"),
        last_spec=None,
        spec=SimpleNamespace(tools=()),
        streamed=False,
    )
    monkeypatch.setattr(terminal, "run_command", _explode)
    monkeypatch.setattr(terminal, "input", _lines(["/boom", ""]), raising=False)
    monkeypatch.setattr(terminal.readline, "read_history_file", lambda *_: None)
    monkeypatch.setattr(terminal.readline, "write_history_file", lambda *_: None)
    monkeypatch.setattr(terminal, "HISTORY_FILE", Path("/nonexistent/root_history"))

    terminal.start_terminal(session)

    printed = capsys.readouterr().out
    assert "/boom failed: RuntimeError: nope" in printed


def _explode(_session, _line):
    raise RuntimeError("nope")


def _lines(sequence):
    remaining = list(sequence)

    def read(_prompt=""):
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return read


def test_reasoning_is_not_streamed_when_the_prompt_opens_the_block():
    written = []
    stream = StreamGate("<|tool_call_start|>", written.append, thinking=True)
    for chunk in ["Let me work ", "it out.\n", "</ifm|think>\n", "The answer ", "is 395."]:
        stream.feed(chunk)
    stream.close()
    assert "".join(written) == "\nThe answer is 395."


def test_a_reasoning_block_opened_mid_stream_is_dropped():
    written = []
    stream = StreamGate("<|tool_call_start|>", written.append)
    for chunk in ["Sure. ", "<think>", "hmm", "</think>", "42"]:
        stream.feed(chunk)
    stream.close()
    assert "".join(written) == "Sure. 42"


def test_an_unclosed_reasoning_block_emits_nothing():
    written = []
    stream = StreamGate("", written.append, thinking=True)
    stream.feed("still reasoning and the tokens ran out")
    stream.close()
    assert written == []


def test_a_closing_tag_split_across_chunks_is_still_found():
    written = []
    stream = StreamGate("", written.append, thinking=True)
    for chunk in ["reasoning", "</ifm", "|think>", "done"]:
        stream.feed(chunk)
    stream.close()
    assert "".join(written) == "done"
