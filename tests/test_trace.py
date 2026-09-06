from root.agent import AgentResult, Step, ToolCall
from root.trace import format_answer, render_step, render_summary, render_turn


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
