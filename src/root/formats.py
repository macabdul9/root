from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

# (name or None, value) per argument: named where the format gives a name,
# positional where it does not.
Arguments = list[tuple[str | None, str]]
Parsed = tuple[str, Arguments] | None

_LFM2_MARKED = re.compile(
    r"<\|tool_call_start\|>\s*\[?(.+?)\]?\s*(?:<\|tool_call_end\|>|$)", re.DOTALL
)
# Models that lose the markers still tend to emit a bare call on its own line.
_BARE_CALL = re.compile(r"^\s*([a-zA-Z_]\w*\(.*\))\s*$", re.MULTILINE)
_LOOSE_CALL = re.compile(r"([a-zA-Z_]\w*)\((.*)\)", re.DOTALL)
_KEYWORD_PREFIX = re.compile(r"^[a-zA-Z_]\w*\s*=\s*")

_QWEN_BLOCK = re.compile(r"<tool_call>\s*(.+?)\s*(?:</tool_call>|$)", re.DOTALL)
_QWEN_FUNCTION = re.compile(r"<function=([a-zA-Z_]\w*)\s*>(.*?)(?:</function>|$)", re.DOTALL)
_QWEN_PARAMETER = re.compile(
    r"<parameter=([a-zA-Z_]\w*)\s*>\s*(.*?)\s*(?:</parameter>|$)", re.DOTALL
)

# A JSON object that is only a tool call, left in the text as if it were prose.
_BARE_JSON_CALL = re.compile(
    r'^\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:.*\}\s*$', re.DOTALL | re.MULTILINE
)

_IFM_BLOCK = re.compile(r"<ifm\|tool_call>\s*(.+?)\s*(?:</ifm\|tool_call>|$)", re.DOTALL)
_IFM_ARG_PAIR = re.compile(
    r"<ifm\|arg_key>\s*(.*?)\s*</ifm\|arg_key>.*?"
    r"<ifm\|arg_value>\s*(.*?)\s*(?:</ifm\|arg_value>|$)",
    re.DOTALL,
)

# Reasoning models wrap their scratchpad in a tag: <think> for Qwen, <ifm|think>
# and its faster variants for K2-Horizon.
_THINKING = re.compile(r"<((?:\w+\|)?think\w*)>.*?</\1>", re.DOTALL)
THINK_OPEN = re.compile(r"<(?:\w+\|)?think\w*>")
THINK_CLOSE = re.compile(r"</(?:\w+\|)?think\w*>")

THINKING_TAGS = (
    "<think>",
    "</think>",
    "<ifm|think>",
    "</ifm|think>",
    "<ifm|think_fast>",
    "</ifm|think_fast>",
    "<ifm|think_faster>",
    "</ifm|think_faster>",
)


def starts_inside_thinking(prompt: str) -> bool:
    """Whether a rendered generation prompt leaves a reasoning block open.

    K2-Horizon's template ends the generation prompt with `<ifm|think>`, so the
    model emits reasoning first and only the closing tag marks the answer. Qwen
    closes the block in the prompt when thinking is disabled, and most models
    have no such block at all.
    """
    opened = list(THINK_OPEN.finditer(prompt))
    if not opened:
        return False
    closed = list(THINK_CLOSE.finditer(prompt))
    return not closed or opened[-1].start() > closed[-1].start()


def strip_calls(text: str, call_format: CallFormat | None = None) -> str:
    """Remove tool-call markup from text being shown as an answer.

    A model that keeps writing after its call, or emits one for a tool that
    does not exist, or is cut off mid-call, leaves the raw markup as the
    answer. Qwen2.5-Coder ended a turn with a bare
    `{"name": "run_python", "arguments": ...}` and another with a truncated
    `<tool_call>` block; neither is something to print at a person.
    """
    markers = set(THINKING_TAGS)
    if call_format is not None:
        markers |= {*call_format.stop, *call_format.keep, call_format.prefill}
    for block in (_LFM2_MARKED, _QWEN_BLOCK, _IFM_BLOCK):
        text = block.sub("", text)
    for marker in sorted(markers, key=len, reverse=True):
        if marker and marker not in THINKING_TAGS:
            text = text.replace(marker, "")
    text = _BARE_JSON_CALL.sub("", text)
    return text.strip()


def strip_thinking(text: str) -> str:
    """Remove a reasoning block from text meant for the user.

    Three shapes have to be handled. Qwen3 emits a matched <think>...</think>
    even with thinking turned off in the template. K2-Horizon's generation
    prompt opens the block, so the model emits only the closing tag and
    everything before it is reasoning. And either can be cut off mid-thought by
    the token limit, leaving an opener with no close.
    """
    text = _THINKING.sub("", text)
    closes = list(THINK_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end() :]
    opened = THINK_OPEN.split(text)
    if len(opened) > 1:
        text = opened[0]
    return text.strip()


def _call_arguments(call: ast.Call) -> Arguments:
    arguments: Arguments = [
        (keyword.arg, str(keyword.value.value))
        for keyword in call.keywords
        if isinstance(keyword.value, ast.Constant)
    ]
    arguments += [(None, str(node.value)) for node in call.args if isinstance(node, ast.Constant)]
    return arguments


def _parse_python_call(source: str) -> Parsed:
    try:
        node = ast.parse(source.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    return node.func.id, _call_arguments(node)


def _parse_python_call_loosely(source: str) -> Parsed:
    """Recover a call whose argument is not valid Python.

    A tool that carries code hits this constantly: the model writes
    `run_python(code="print(len("x"))")` and the nested quotes make the whole
    call unparseable, even though what it meant is unambiguous.
    """
    match = _LOOSE_CALL.fullmatch(source.strip())
    if match is None:
        return None
    argument = _KEYWORD_PREFIX.sub("", match.group(2).strip())
    if len(argument) >= 2 and argument[0] == argument[-1] and argument[0] in "\"'":
        argument = argument[1:-1]
    argument = argument.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
    return match.group(1), [(None, argument)]


def _parse_bare_call(text: str, tool_names: set[str]) -> Parsed:
    for match in _BARE_CALL.finditer(text):
        parsed = _parse_python_call(match.group(1)) or _parse_python_call_loosely(match.group(1))
        if parsed and parsed[0] in tool_names:
            return parsed
    return None


def parse_lfm2(text: str, tool_names: set[str]) -> Parsed:
    marked = _LFM2_MARKED.search(text)
    if marked:
        parsed = _parse_python_call(marked.group(1))
        if parsed:
            return parsed
        loose = _parse_python_call_loosely(marked.group(1))
        if loose and loose[0] in tool_names:
            return loose
    return _parse_bare_call(text, tool_names)


def _json_arguments(payload: dict) -> Arguments:
    arguments = payload.get("arguments") or {}
    if isinstance(arguments, str):
        return [(None, arguments)]
    return [(str(key), str(value)) for key, value in arguments.items()]


def parse_qwen_json(text: str, tool_names: set[str]) -> Parsed:
    parsed = _parse_qwen_blocks(text)
    if parsed:
        return parsed
    # Qwen2.5-Coder writes the object without a closing tag and then keeps
    # going, so the block match runs to the end of the text and no longer
    # parses. Scanning for the first balanced object finds the call in that,
    # in a bare object with no wrapper at all, and in one followed by prose.
    # An unrecognised call becomes an error the model can act on, where
    # ignoring it leaves the turn with no answer at all.
    payload = _first_json_object(text)
    if isinstance(payload, dict) and "name" in payload:
        return str(payload["name"]), _json_arguments(payload)
    return _parse_bare_call(text, tool_names)


def _first_json_object(text: str) -> dict | None:
    """The first balanced {...} in the text, parsed, or None.

    Brace counting rather than a regular expression, because the arguments are
    nested objects and quotes may hold braces of their own.
    """
    for start in (index for index, char in enumerate(text) if char == "{"):
        depth, in_string, escaped = 0, False, False
        for position in range(start, len(text)):
            char = text[position]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = not in_string
            elif not in_string and char == "{":
                depth += 1
            elif not in_string and char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        found = json.loads(text[start : position + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(found, dict) and "name" in found:
                        return found
                    break
    return None


def _parse_qwen_blocks(text: str) -> Parsed:
    for match in _QWEN_BLOCK.finditer(text):
        body = match.group(1)
        # A truncated block loses its closing brace; adding one back is safe
        # because the model only ever emits a single flat object here.
        for candidate in (body, body + "}", body + '"}}'):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "name" in payload:
                return str(payload["name"]), _json_arguments(payload)
    return None


def parse_qwen_xml(text: str, tool_names: set[str]) -> Parsed:
    function = _QWEN_FUNCTION.search(text)
    if function is None:
        return _parse_bare_call(text, tool_names)
    arguments = [
        (name, value.strip()) for name, value in _QWEN_PARAMETER.findall(function.group(2))
    ]
    return function.group(1), arguments


def parse_ifm(text: str, tool_names: set[str]) -> Parsed:
    block = _IFM_BLOCK.search(text)
    if block is None:
        return _parse_bare_call(text, tool_names)
    body = block.group(1)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and "name" in payload:
        return str(payload["name"]), _json_arguments(payload)

    name = body.splitlines()[0].strip()
    return name, [(key, value) for key, value in _IFM_ARG_PAIR.findall(body)]


# What a value may contain, per format: whatever cannot terminate the field.
# Empty is allowed everywhere, because a tool can take an empty argument -
# `today` with no offset means now, and forbidding it makes the model invent
# one. The quoted form accepts either delimiter, so code holding an apostrophe
# can be written inside double quotes rather than escaped into illegibility.
# Unbounded rather than length-capped: a counted repetition like {0,600} makes
# the compiler unroll 600 states per position, and across an alternation of
# tools that overflows the DFA and the grammar silently falls back. Length is
# already bounded by max_new_tokens.
_SINGLE = r"'([^'\\]|\\.)*'"
_DOUBLE = r"\"([^\"\\]|\\.)*\""
_QUOTED = f"({_SINGLE}|{_DOUBLE})"
_JSON_STRING = r"([^\"\\]|\\.)*"
_UNTIL_TAG = r"[^<]*"

Signature = tuple[str, tuple[str, ...]]


def _alternatives(parts: list[str]) -> str:
    return "(" + "|".join(parts) + ")"


def regex_lfm2(signatures: list[Signature]) -> str:
    calls = [
        name + r"\(" + r", ".join(f"{p}={_QUOTED}" for p in params) + r"\)"
        for name, params in signatures
    ]
    return _alternatives(calls) + r"\]<\|tool_call_end\|>"


def regex_qwen_json(signatures: list[Signature]) -> str:
    # The prefill already opened `{"name": "`, so the name comes first and bare.
    calls = [
        name
        + r'", "arguments": \{'
        + r", ".join(f'"{p}": "{_JSON_STRING}"' for p in params)
        + r"\}\}"
        for name, params in signatures
    ]
    return _alternatives(calls) + r"\n?</tool_call>"


def regex_qwen_xml(signatures: list[Signature]) -> str:
    calls = [
        name
        + r">\n"
        + "".join(f"<parameter={p}>\n{_UNTIL_TAG}\n</parameter>\n" for p in params)
        + r"</function>\n</tool_call>"
        for name, params in signatures
    ]
    return _alternatives(calls)


def regex_ifm(signatures: list[Signature]) -> str:
    calls = [
        name
        + r"\n"
        + "".join(
            f"<ifm\\|arg_key>{p}</ifm\\|arg_key>\\n"
            f"<ifm\\|arg_value>{_UNTIL_TAG}</ifm\\|arg_value>\\n"
            for p in params
        )
        + r"</ifm\|tool_call>\n</ifm\|tool_calls>"
        for name, params in signatures
    ]
    return _alternatives(calls)


@dataclass(frozen=True, slots=True)
class CallFormat:
    """How one model family writes a tool call.

    `prefill` opens an assistant turn so a small model has to complete a call
    rather than answering from memory; an empty prefill means the format cannot
    be forced that way.
    """

    name: str
    marker: str
    prefill: str
    stop: tuple[str, ...]
    keep: tuple[str, ...]
    parse: Callable[[str, set[str]], Parsed]
    regex: Callable[[list[Signature]], str] | None = None

    @property
    def forceable(self) -> bool:
        return bool(self.prefill)

    @property
    def protected(self) -> str:
        """Every marker that must survive decoding for `parse` to work.

        Most of these are control tokens in the tokenizer's vocabulary, and
        stripping them along with the rest leaves a call the parser cannot read.
        """
        return "".join([self.prefill, *self.stop, *self.keep, *THINKING_TAGS])


LFM2 = CallFormat(
    name="lfm2",
    marker="<|tool_call_start|>",
    prefill="<|tool_call_start|>[",
    stop=("<|tool_call_end|>",),
    keep=("<|tool_call_start|>",),
    parse=parse_lfm2,
    regex=regex_lfm2,
)
QWEN_XML = CallFormat(
    name="qwen-xml",
    marker="<function=",
    prefill="<tool_call>\n<function=",
    stop=("</tool_call>",),
    keep=("<tool_call>", "<function=", "</function>", "<parameter=", "</parameter>"),
    parse=parse_qwen_xml,
    regex=regex_qwen_xml,
)
QWEN_JSON = CallFormat(
    name="qwen-json",
    marker="<tool_call>",
    prefill='<tool_call>\n{"name": "',
    stop=("</tool_call>",),
    keep=("<tool_call>",),
    parse=parse_qwen_json,
    regex=regex_qwen_json,
)
IFM = CallFormat(
    name="ifm",
    marker="<ifm|tool_call>",
    prefill="<ifm|tool_calls>\n<ifm|tool_call>",
    stop=("</ifm|tool_calls>", "</ifm|tool_call>"),
    keep=(
        "<ifm|tool_calls>",
        "<ifm|tool_call>",
        "<ifm|arg_key>",
        "</ifm|arg_key>",
        "<ifm|arg_type>",
        "</ifm|arg_type>",
        "<ifm|arg_value>",
        "</ifm|arg_value>",
    ),
    parse=parse_ifm,
    regex=regex_ifm,
)
GENERIC = CallFormat(
    name="generic",
    marker="",
    prefill="",
    stop=(),
    keep=(),
    parse=_parse_bare_call,
)

# Ordered: qwen-xml before qwen-json, since an XML template contains both markers.
KNOWN = (IFM, LFM2, QWEN_XML, QWEN_JSON)


def detect_format(chat_template: str) -> CallFormat:
    """Pick the call format from the model's own chat template.

    Reading the template beats a per-model table: a model the project has never
    heard of still gets the right format if it writes calls like one it has.
    """
    for candidate in KNOWN:
        if candidate.marker in chat_template:
            return candidate
    return GENERIC
