import re

import pytest

from root.formats import (
    GENERIC,
    IFM,
    LFM2,
    QWEN_JSON,
    QWEN_XML,
    THINKING_TAGS,
    detect_format,
    starts_inside_thinking,
    strip_thinking,
)

NAMES = {"calculator", "run_python", "write_file"}


def called(parsed):
    """Name and argument values, dropping parameter names.

    Every format names its arguments differently, or not at all for a bare
    call, so the values are what a cross-format assertion can compare.
    """
    name, arguments = parsed
    return name, [value for _, value in arguments]


def test_each_family_is_detected_from_its_template():
    assert detect_format("...<|tool_call_start|>...").name == "lfm2"
    assert detect_format('...<tool_call>\n{"name":...').name == "qwen-json"
    assert detect_format("...<tool_call>\n<function=x>...").name == "qwen-xml"
    assert detect_format("...<ifm|tool_call>...").name == "ifm"


def test_an_unknown_template_falls_back_to_the_generic_format():
    fmt = detect_format("a template with no tool markers at all")
    assert fmt is GENERIC
    assert not fmt.forceable


LFM2_CALL = "<|tool_call_start|>[calculator(expression='6 * 7')]<|tool_call_end|>"
QWEN_JSON_CALL = (
    '<tool_call>\n{"name": "calculator", "arguments": {"expression": "6 * 7"}}\n</tool_call>'
)
QWEN_XML_CALL = (
    "<tool_call>\n<function=calculator>\n"
    "<parameter=expression>\n6 * 7\n</parameter>\n</function>\n</tool_call>"
)
IFM_CALL = (
    "<ifm|tool_calls>\n<ifm|tool_call>"
    '{"name": "calculator", "arguments": {"expression": "6 * 7"}}'
    "</ifm|tool_call>\n</ifm|tool_calls>"
)


@pytest.mark.parametrize(
    "call_format,text",
    [
        (LFM2, LFM2_CALL),
        (QWEN_JSON, QWEN_JSON_CALL),
        (QWEN_XML, QWEN_XML_CALL),
        (IFM, IFM_CALL),
    ],
)
def test_every_format_reads_its_own_call(call_format, text):
    assert called(call_format.parse(text, NAMES)) == ("calculator", ["6 * 7"])


def test_ifm_reads_the_key_value_form():
    text = (
        "<ifm|tool_calls>\n<ifm|tool_call>calculator\n"
        "<ifm|arg_key>expression</ifm|arg_key>\n"
        "<ifm|arg_value>6 * 7</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>"
    )
    assert IFM.parse(text, NAMES) == ("calculator", [("expression", "6 * 7")])


def test_qwen_json_recovers_a_block_cut_off_by_the_token_limit():
    text = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "6 * 7"}'
    assert called(QWEN_JSON.parse(text, NAMES)) == ("calculator", ["6 * 7"])


def test_lfm2_recovers_a_call_with_nested_quotes():
    text = '<|tool_call_start|>[run_python(code="print(len("abc"))")]<|tool_call_end|>'
    assert called(LFM2.parse(text, NAMES)) == ("run_python", ['print(len("abc"))'])


def test_lfm2_unescapes_newlines_in_a_recovered_call():
    text = r'<|tool_call_start|>[run_python(code="x = 1\nprint(\'x\')")]'
    name, [code] = called(LFM2.parse(text, NAMES))
    assert name == "run_python"
    assert code.splitlines() == ["x = 1", "print('x')"]


@pytest.mark.parametrize("call_format", [LFM2, QWEN_JSON, QWEN_XML, IFM, GENERIC])
def test_every_format_falls_back_to_a_bare_call(call_format):
    assert called(call_format.parse("calculator('2 + 2')", NAMES)) == ("calculator", ["2 + 2"])


@pytest.mark.parametrize("call_format", [LFM2, QWEN_JSON, QWEN_XML, IFM, GENERIC])
def test_no_format_finds_a_call_in_prose(call_format):
    assert call_format.parse("The answer is 42.", NAMES) is None


@pytest.mark.parametrize("call_format", [LFM2, QWEN_JSON, QWEN_XML, IFM, GENERIC])
def test_no_format_invents_an_unknown_tool(call_format):
    assert call_format.parse("print('hello')", NAMES) is None


@pytest.mark.parametrize("call_format", [LFM2, QWEN_JSON, QWEN_XML, IFM])
def test_a_forceable_prefill_opens_the_format_it_belongs_to(call_format):
    assert call_format.forceable
    assert call_format.marker in call_format.prefill


def test_thinking_is_removed_from_an_answer():
    assert strip_thinking("<think>weighing it up</think>\nThe answer is 4.") == "The answer is 4."


def test_an_unclosed_thinking_block_keeps_what_came_before():
    assert strip_thinking("Here goes. <think>still weighing") == "Here goes."


def test_text_without_thinking_is_untouched():
    assert strip_thinking("The answer is 4.") == "The answer is 4."


def test_a_vendor_thinking_tag_is_removed_too():
    text = "<ifm|think>let me work through it</ifm|think>\nThe answer is 4."
    assert strip_thinking(text) == "The answer is 4."


def test_every_thinking_tag_is_protected_from_special_token_stripping():
    for tag in THINKING_TAGS:
        assert strip_thinking(f"<think>x</think>{tag}") != tag


def test_reasoning_before_a_lone_closing_tag_is_dropped():
    text = "The user asks about caches. Let me think.\n</ifm|think>\nA KV cache stores keys."
    assert strip_thinking(text) == "A KV cache stores keys."


def test_the_last_closing_tag_wins():
    assert strip_thinking("a</think>b</think>final") == "final"


@pytest.mark.parametrize(
    "call_format,text",
    [
        (LFM2, LFM2_CALL),
        (QWEN_JSON, QWEN_JSON_CALL),
        (QWEN_XML, QWEN_XML_CALL),
        (IFM, IFM_CALL),
    ],
)
def test_a_format_protects_every_marker_its_parser_needs(call_format, text):
    """Decoding drops control tokens; a marker left out of `protected` is one
    the parser will never see, which reads as the model not calling anything."""
    stripped = text
    for marker in set(re.findall(r"</?[\w|]+[=>]", text)):
        if marker not in call_format.protected:
            stripped = stripped.replace(marker, "")
    assert called(call_format.parse(stripped, NAMES)) == ("calculator", ["6 * 7"])


TWO_ARGUMENT_CALLS = [
    (LFM2, "<|tool_call_start|>[write_file(path='a.txt', content='hi')]<|tool_call_end|>"),
    (
        QWEN_JSON,
        '<tool_call>{"name": "write_file", "arguments": {"path": "a.txt", "content": "hi"}}'
        "</tool_call>",
    ),
    (
        QWEN_XML,
        "<tool_call><function=write_file><parameter=path>a.txt</parameter>"
        "<parameter=content>hi</parameter></function></tool_call>",
    ),
    (
        IFM,
        "<ifm|tool_call>write_file\n<ifm|arg_key>path</ifm|arg_key>\n"
        "<ifm|arg_value>a.txt</ifm|arg_value>\n<ifm|arg_key>content</ifm|arg_key>\n"
        "<ifm|arg_value>hi</ifm|arg_value></ifm|tool_call>",
    ),
]


@pytest.mark.parametrize("call_format,text", TWO_ARGUMENT_CALLS)
def test_every_format_reads_a_two_argument_call(call_format, text):
    name, arguments = call_format.parse(text, NAMES)
    assert name == "write_file"
    assert dict(arguments) == {"path": "a.txt", "content": "hi"}


def test_a_positional_two_argument_call_keeps_its_order():
    parsed = LFM2.parse("write_file('a.txt', 'hi')", NAMES)
    assert called(parsed) == ("write_file", ["a.txt", "hi"])


def test_a_prompt_that_opens_a_reasoning_block_is_detected():
    assert starts_inside_thinking("<|im_start|>assistant\n<ifm|think>\n")


def test_a_prompt_that_closes_it_again_is_not():
    assert not starts_inside_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_a_prompt_without_reasoning_is_not():
    assert not starts_inside_thinking("<|im_start|>assistant\n")


CALL_SAMPLES = {
    "lfm2": (LFM2, "calculator(expression='6 * 7')]<|tool_call_end|>"),
    "qwen-json": (
        QWEN_JSON,
        'calculator", "arguments": {"expression": "6 * 7"}}\n</tool_call>',
    ),
    "qwen-xml": (
        QWEN_XML,
        "calculator>\n<parameter=expression>\n6 * 7\n</parameter>\n</function>\n</tool_call>",
    ),
    "ifm": (
        IFM,
        "calculator\n<ifm|arg_key>expression</ifm|arg_key>\n"
        "<ifm|arg_value>6 * 7</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
    ),
}
SIGNATURES = [("calculator", ("expression",)), ("write_file", ("path", "content"))]


@pytest.mark.parametrize("name", list(CALL_SAMPLES))
def test_every_format_regex_matches_a_real_call(name):
    call_format, sample = CALL_SAMPLES[name]
    assert re.compile(call_format.regex(SIGNATURES), re.DOTALL).fullmatch(sample)


def test_the_regex_accepts_either_quote_style():
    """Code holding an apostrophe has to be writable in double quotes; forcing
    single quotes measurably degrades what the model writes."""
    pattern = re.compile(LFM2.regex([("run_python", ("code",))]), re.DOTALL)
    assert pattern.fullmatch("""run_python(code='print(1)')]<|tool_call_end|>""")
    assert pattern.fullmatch("""run_python(code="print('x')")]<|tool_call_end|>""")


def test_the_regex_accepts_an_empty_argument():
    """today() with no offset means now; forbidding empty makes one up."""
    pattern = re.compile(LFM2.regex([("today", ("offset",))]), re.DOTALL)
    assert pattern.fullmatch("today(offset='')]<|tool_call_end|>")


def test_the_regex_is_unbounded_rather_than_counted():
    """A counted repetition unrolls per position and overflows the DFA once
    several tools are in the alternation, which silently disables the grammar."""
    assert "{0," not in LFM2.regex(SIGNATURES)
    assert "{1," not in LFM2.regex(SIGNATURES)


def test_a_call_with_no_closing_tag_and_trailing_junk_is_still_read():
    """Qwen2.5-Coder writes the object, omits </tool_call>, then keeps going.
    The block match then runs to the end of the text and no longer parses."""
    text = (
        '<tool_call> {"name": "dijkstra", "arguments": {"graph": {"nodes": ["A"]}}} '
        '```json { "name": "dijkstra"'
    )
    name, arguments = QWEN_JSON.parse(text, NAMES)

    assert name == "dijkstra"
    assert dict(arguments)["graph"]


def test_a_bare_json_object_is_read_as_a_call():
    text = '{"name": "run_python", "arguments": {"code": "print(1)"}}'
    assert QWEN_JSON.parse(text, NAMES) == ("run_python", [("code", "print(1)")])


def test_braces_inside_a_string_do_not_confuse_the_scan():
    text = '{"name": "run_python", "arguments": {"code": "print({\'a\': 1})"}}'
    assert QWEN_JSON.parse(text, NAMES) == ("run_python", [("code", "print({'a': 1})")])


def test_prose_holding_a_brace_is_not_a_call():
    assert QWEN_JSON.parse("Use a dict like {'a': 1} for that.", NAMES) is None
