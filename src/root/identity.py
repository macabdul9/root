from __future__ import annotations

from .config import AgentSpec
from .tools import Tool

NAME = "root"
AUTHOR = "Abdul Waheed"

# Grouped by what a person would ask for, not by tool name, and built from the
# registry so it cannot drift from what is actually installed.
ABILITIES = (
    ("calculator", "do arithmetic exactly"),
    ("run_python", "write and run a short Python program"),
    ("read_file", "read files in your workspace"),
    ("write_file", "create and edit files"),
    ("read_pdf", "pull the text out of a PDF"),
    ("grep", "search your code"),
    ("csv_query", "answer questions about a CSV in SQL"),
    ("git", "read your git history"),
    ("today", "tell you the date"),
)


def describe(model_id: str, tools: dict[str, Tool], agents: dict[str, AgentSpec]) -> str:
    """Who root is, in its own words, with nothing invented.

    The model name and the ability list come from what is actually loaded, so
    this stays true when the model is switched or a tool is added.
    """
    can = [phrase for name, phrase in ABILITIES if name in tools]
    abilities = ", ".join(can[:-1]) + f", and {can[-1]}" if len(can) > 1 else "".join(can)
    routes = ", ".join(sorted(agents))
    return (
        f'I am an AI agent named "{NAME}" (no, I am not scary) built by {AUTHOR}. '
        f"I live in your terminal. The underlying large language model (or rather smol) "
        f"is {model_id}.\n\n"
        f"I can {abilities}. I do that by routing what you ask to one of a few small "
        f"agents ({routes}), each of which gets only the tools it needs.\n\n"
        f"Type /help for the commands, /agents to see the routing, and /tools for what "
        f"the current agent can call."
    )
