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

_IFM_BLOCK = re.compile(r"<ifm\|tool_call>\s*(.+?)\s*(?:</ifm\|tool_call>|$)", re.DOTALL)
_IFM_ARG_PAIR = re.compile(
    r"<ifm\|arg_key>\s*(.*?)\s*</ifm\|arg_key>.*?"
    r"<ifm\|arg_value>\s*(.*?)\s*(?:</ifm\|arg_value>|$)",
    re.DOTALL,
)

# Reasoning models wrap their scratchpad in a tag: <think> for Qwen, <ifm|think>
# and its faster variants for K2-Horizon.
_THINKING = re.compile(r"<((?:\w+\|)?think\w*)>.*?</\1>", re.DOTALL)
_THINK_OPEN = re.compile(r"<(?:\w+\|)?think\w*>")
_THINK_CLOSE = re.compile(r"</(?:\w+\|)?think\w*>")

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


def strip_thinking(text: str) -> str:
    """Remove a reasoning block from text meant for the user.

    Three shapes have to be handled. Qwen3 emits a matched <think>...</think>
    even with thinking turned off in the template. K2-Horizon's generation
    prompt opens the block, so the model emits only the closing tag and
    everything before it is reasoning. And either can be cut off mid-thought by
    the token limit, leaving an opener with no close.
    """
    text = _THINKING.sub("", text)
    closes = list(_THINK_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end() :]
    opened = _THINK_OPEN.split(text)
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
    return _parse_bare_call(text, tool_names)


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
)
QWEN_XML = CallFormat(
    name="qwen-xml",
    marker="<function=",
    prefill="<tool_call>\n<function=",
    stop=("</tool_call>",),
    keep=("<tool_call>", "<function=", "</function>", "<parameter=", "</parameter>"),
    parse=parse_qwen_xml,
)
QWEN_JSON = CallFormat(
    name="qwen-json",
    marker="<tool_call>",
    prefill='<tool_call>\n{"name": "',
    stop=("</tool_call>",),
    keep=("<tool_call>",),
    parse=parse_qwen_json,
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
