from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_AGENT = "chat"
AUTO = "auto"

# Prompts that are about the assistant rather than a task. A small model told to
# call a tool will call one for these too, and `what can you do?` answered with
# run_python('print(time.time())') is worse than no tool at all.
_CONVERSATIONAL = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|yep|nope)\b"
    r"|\b(what|which) (can|could) you (do|help)"
    r"|\bwho are you\b|\bwhat are you\b|\bhow do you work\b"
    r"|\bwhat (tools|agents|models) (do you|are)\b",
    re.IGNORECASE,
)


def is_conversational(prompt: str) -> bool:
    """True for prompts about the assistant itself, which no tool can answer."""
    return bool(_CONVERSATIONAL.search(prompt))


# Ordered: the first match wins, so narrow rules come before broad ones. These
# catch a little over half of tool-worthy prompts on phrasings they were not
# written against; the rest fall through to the default agent, which is what
# would have happened anyway. Never let a rule fire on a prompt it is unsure
# about - a miss costs nothing, a wrong route runs the wrong tool.
RULES: tuple[tuple[str, str, str], ...] = (
    (
        "write code",
        r"\b(write|show|give|generate)\b.{0,20}\b(code|function|program|script|class|snippet)\b"
        r"|\bimplement\b.{0,25}\b(function|algorithm|class)\b",
        "code",
    ),
    (
        "create a file",
        r"\b(create|write|save|make|append to)\b.{0,25}"
        r"\b(files?|director(y|ies)|folders?|\w+\.\w{1,4})\b",
        "write",
    ),
    (
        "code reference",
        r"\b(where|which|what)\b.{0,25}\b(file|module|function|class|method|script)\b"
        r".{0,40}\b(defin|declar|implement|import|use|call|live|set)",
        "search",
    ),
    (
        "search verb",
        r"\b(grep|search for|look for|find (all|every|the) (use|call|reference))\b",
        "search",
    ),
    ("path", r"\b[\w./-]+\.(py|md|toml|ya?ml|json|txt|sh|csv|cfg|ini|lock)\b", "files"),
    (
        "filesystem",
        r"\b(file|files|directory|folder|repo|repository|workspace|codebase)\b"
        r"|\b(list|show)\b.{0,25}\b(under|inside|in)\b",
        "files",
    ),
    (
        "record",
        r"\b(invoice|receipt|statement|purchase order|order confirmation|bill from)\b"
        r"|\bextract\b.{0,40}\b(field|json|name|date|amount)\b"
        r"|\b\d{4}-\d{2}-\d{2}\b.{0,40}\d+\.\d{2}\b",
        "extract",
    ),
    (
        "clock",
        r"\bwhat time\b|\bcurrent (time|date)\b|\btoday'?s date\b"
        r"|\b(what|which)\b.{0,15}\b(time|date|day)\b.{0,12}\b(is it|is today)\b",
        "now",
    ),
    ("list literal", r"\[[^\]]*\d[^\]]*\]", "python"),
    (
        "data verb",
        r"\b(sort|reverse|shuffle|dedup|deduplicate|median|average|mean|standard deviation"
        r"|variance|percentile|histogram|parse|regex|convert|round|count|trim)\b",
        "python",
    ),
    (
        "counting",
        r"\bhow many\b.{0,30}\b(character|letter|word|line|item|element|day|week|month|year|vowel)",
        "python",
    ),
    (
        "dates",
        r"\b\d{4}-\d{2}-\d{2}\b"
        r"|\b(date|day|days|week|month|year)\b.{0,30}\b(between|until|since|apart|ago|add|subtract)\b",
        "python",
    ),
    (
        "arithmetic",
        r"\d\s*[-+*/^%]\s*\d|\b\d+\s*%"
        r"|\b\d+\b.{0,25}\b(times|plus|minus|divided by|multiplied by|percent|squared?)\b",
        "calc",
    ),
    (
        "word problem",
        r"\b(how many|how much|what is the (sum|total|product|difference|average))\b",
        "calc",
    ),
)

_COMPILED = tuple(
    (name, re.compile(pattern, re.IGNORECASE), agent) for name, pattern, agent in RULES
)


@dataclass(frozen=True, slots=True)
class Route:
    agent: str
    rule: str | None

    @property
    def matched(self) -> bool:
        return self.rule is not None


def choose_agent(prompt: str, available: set[str], default: str = DEFAULT_AGENT) -> Route:
    """Pick the agent for a prompt from its wording alone.

    A 350M model cannot do this classification - asked to name a handler it
    replies "search" for `17 times 23` - so the choice is made from rules that
    are at least inspectable and testable.
    """
    for name, pattern, agent in _COMPILED:
        if agent in available and pattern.search(prompt):
            return Route(agent=agent, rule=name)
    return Route(agent=default if default in available else sorted(available)[0], rule=None)
