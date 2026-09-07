from __future__ import annotations

import logging
import os
from typing import Any

from .formats import CallFormat, Signature

logger = logging.getLogger(__name__)

BACKEND = "outlines_core"
# Each compiled index is held for reuse, but they are large: one per agent over
# a big vocabulary is enough to get a run killed. Keep only the most recent few.
MAX_CACHED = 2
_processors: dict[tuple[int, str], Any] = {}
_wrapped: dict[int, Any] = {}
_unavailable = False


def signatures_from(schemas: list[dict] | None) -> list[Signature] | None:
    """The (name, parameter names) of each tool, from its JSON schema.

    None when any tool takes a free-form value, because the grammar covers the
    whole generation: one unconstrainable branch means the turn cannot be
    constrained at all.
    """
    pairs = []
    for schema in schemas or []:
        function = schema.get("function", schema)
        properties = (function.get("parameters") or {}).get("properties") or {}
        if any(field.get("x-freeform") for field in properties.values()):
            return None
        pairs.append((function["name"], tuple(properties)))
    return pairs


def call_processor(model, tokenizer, call_format: CallFormat, schemas: list[dict] | None):
    """A logits processor that only allows a well-formed tool call.

    Returns None when the format has no grammar, when there are no tools, or
    when outlines is not installed - in every case generation runs unconstrained
    and the parser's repairs pick up the slack.
    """
    global _unavailable
    if _unavailable or call_format.regex is None:
        return None
    if os.environ.get("ROOT_NO_GRAMMAR"):
        return None
    pairs = signatures_from(schemas)
    if not pairs:
        return None

    pattern = call_format.regex(pairs)
    key = (id(model), pattern)
    if key in _processors:
        return _reset(_processors[key])

    try:
        import outlines
        from outlines.backends import get_regex_logits_processor
    except ImportError:
        # Optional: `uv sync --extra grammar` turns constrained decoding on.
        logger.info("outlines is not installed; tool calls are parsed rather than constrained")
        _unavailable = True
        return None

    try:
        steerable = _wrapped.get(id(model))
        if steerable is None:
            steerable = _wrapped[id(model)] = outlines.from_transformers(model, tokenizer)
        while len(_processors) >= MAX_CACHED:
            _processors.pop(next(iter(_processors)))
        _processors[key] = get_regex_logits_processor(BACKEND, steerable, pattern)
    except Exception:
        # A grammar that will not compile is not worth failing a run over.
        logger.warning("could not build a tool-call grammar; falling back", exc_info=True)
        _processors[key] = None
    return _reset(_processors[key])


def _reset(processor):
    """Clear the guide state left behind by the previous generation.

    Compiling the index is the expensive part and is worth caching, but the
    processor walks an FSM per sequence: reused without a reset, the second
    call starts mid-state and raises on the first token it does not expect.
    """
    if processor is not None:
        processor.reset()
    return processor
