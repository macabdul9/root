from __future__ import annotations

import ast
import json
import operator
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

MAX_TOOL_OUTPUT_CHARS = 800
MAX_LISTED_FILES = 40
MAX_GREP_MATCHES = 30
MAX_GREP_FILE_BYTES = 1_000_000
PYTHON_TIMEOUT_SECONDS = 10

SKIPPED_DIRS = frozenset(
    {".git", ".venv", "venv", "__pycache__", "runs", ".pytest_cache", ".ruff_cache", "node_modules"}
)

_OFFSET = re.compile(r"^([+-]?\d+)\s*(second|minute|hour|day|week)s?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool the model calls with exactly one string argument.

    One argument keeps both the JSON schema and the parsing of the model's call
    small enough that a sub-billion-parameter model gets it right.
    """

    name: str
    description: str
    parameter: str
    parameter_description: str
    run: Callable[[str], str]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        self.parameter: {
                            "type": "string",
                            "description": self.parameter_description,
                        }
                    },
                    "required": [self.parameter],
                },
            },
        }


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression without exposing eval to model output.

    Walks the parsed tree by hand, so names, calls, and attribute access raise
    rather than executing.
    """
    expression = expression.strip().rstrip("=").strip()
    if not expression:
        raise ValueError("empty expression")
    value = _eval(ast.parse(expression, mode="eval").body)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(round(value, 6) if isinstance(value, float) else value)


def today(offset: str) -> str:
    """Current local date and time, optionally shifted by an offset.

    The model has no clock and will invent a date if asked for one, so this is
    the only trustworthy source of "now" in the loop.
    """
    offset = offset.strip().replace("in ", "").replace(" ago", "")
    moment = datetime.now()
    if offset and offset.lower() not in {"now", "today"}:
        match = _OFFSET.match(offset)
        if match is None:
            raise ValueError(f"offset must look like '+3 days' or '-2 weeks', got {offset!r}")
        amount, unit = int(match.group(1)), match.group(2).lower()
        moment += timedelta(**{f"{unit}s": amount})
    return moment.strftime("%Y-%m-%d %H:%M:%S (%A)")


def ensure_print(code: str) -> str:
    """Wrap a bare expression in print().

    Small models write `sum(range(1, 21))` with no print, get nothing back, and
    then invent a number rather than retrying.
    """
    if "print(" in code:
        return code
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return code
    return f"print({ast.unparse(tree.body)})"


def _dig(document: object, dotted: str) -> object:
    value = document
    for key in dotted.split("."):
        if isinstance(value, list):
            value = value[int(key)]
        elif isinstance(value, dict):
            if key not in value:
                raise KeyError(f"no key {key!r} in {sorted(value)}")
            value = value[key]
        else:
            raise KeyError(f"cannot index {type(value).__name__} with {key!r}")
    return value


def _resolve_inside(workspace: Path, raw: str) -> Path:
    candidate = (workspace / raw.strip().strip("'\"")).resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError(f"path is outside the workspace: {raw.strip()}")
    return candidate


def build_tools(workspace: Path) -> dict[str, Tool]:
    workspace = workspace.resolve()

    def read_file(argument: str) -> str:
        path = _resolve_inside(workspace, argument)
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {path.relative_to(workspace)}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            return text[:MAX_TOOL_OUTPUT_CHARS] + "\n[truncated]"
        return text

    def list_files(argument: str) -> str:
        pattern = argument.strip() or "*"
        matches = sorted(
            str(p.relative_to(workspace)) for p in workspace.glob(pattern) if p.is_file()
        )
        if not matches:
            return f"no files match {pattern}"
        listed = matches[:MAX_LISTED_FILES]
        if len(matches) > len(listed):
            listed.append(f"[{len(matches) - len(listed)} more]")
        return "\n".join(listed)

    def grep(argument: str) -> str:
        pattern = argument.strip()
        if not pattern:
            raise ValueError("empty search pattern")
        try:
            expression = re.compile(pattern)
        except re.error:
            expression = re.compile(re.escape(pattern))

        matches: list[str] = []
        for root, dirnames, filenames in os.walk(workspace):
            dirnames[:] = sorted(
                d for d in dirnames if d not in SKIPPED_DIRS and not d.startswith(".")
            )
            for filename in sorted(filenames):
                path = Path(root) / filename
                if path.stat().st_size > MAX_GREP_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                if "\x00" in text[:1024]:
                    continue
                for number, line in enumerate(text.splitlines(), start=1):
                    if expression.search(line):
                        relative = path.relative_to(workspace)
                        matches.append(f"{relative}:{number}: {line.strip()[:160]}")
                        if len(matches) >= MAX_GREP_MATCHES:
                            return "\n".join(matches) + "\n[more matches not shown]"
        return "\n".join(matches) if matches else f"no matches for {pattern}"

    def file_info(argument: str) -> str:
        path = _resolve_inside(workspace, argument)
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {argument.strip()}")
        stat = path.stat()
        lines = path.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        return (
            f"{path.relative_to(workspace)}: {lines} lines, "
            f"{stat.st_size} bytes, modified {modified}"
        )

    def json_get(argument: str) -> str:
        if ":" not in argument:
            raise ValueError("expected 'file.json:dotted.path', for example config.json:model.name")
        raw_path, dotted = argument.split(":", 1)
        path = _resolve_inside(workspace, raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {raw_path.strip()}")
        value = _dig(json.loads(path.read_text(encoding="utf-8")), dotted.strip())
        rendered = value if isinstance(value, str) else json.dumps(value)
        return rendered[:MAX_TOOL_OUTPUT_CHARS]

    def run_python(code: str) -> str:
        """Execute model-written Python in a child process and return what it printed.

        The child is isolated from the parent interpreter and killed at
        PYTHON_TIMEOUT_SECONDS, but this is not a security sandbox: the code runs
        as the current user with the workspace as its working directory.
        """
        code = ensure_print(code.strip())
        if not code:
            raise ValueError("empty program")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=PYTHON_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"program did not finish in {PYTHON_TIMEOUT_SECONDS}s") from None
        output = (completed.stdout + completed.stderr).strip()
        if not output:
            raise ValueError(
                "the program printed nothing; call again with print(...) around the result"
            )
        if len(output) > MAX_TOOL_OUTPUT_CHARS:
            output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n[truncated]"
        # A crashed program is an error the model has to act on, not a result.
        return output if completed.returncode == 0 else f"error: the program failed\n{output}"

    return {
        "calculator": Tool(
            name="calculator",
            description="Evaluate one arithmetic expression and return the number.",
            parameter="expression",
            parameter_description="Arithmetic over numbers, such as 24 * 7 - 13.",
            run=calculate,
        ),
        "read_file": Tool(
            name="read_file",
            description="Read a text file from the workspace and return its contents.",
            parameter="path",
            parameter_description="Path relative to the workspace, such as src/app.py.",
            run=read_file,
        ),
        "list_files": Tool(
            name="list_files",
            description="List workspace files matching a glob pattern.",
            parameter="pattern",
            parameter_description="Glob pattern such as *.py or src/**/*.md.",
            run=list_files,
        ),
        "grep": Tool(
            name="grep",
            description="Search every workspace file for a pattern and return matching lines.",
            parameter="pattern",
            parameter_description="Text or regular expression, such as force_first_call.",
            run=grep,
        ),
        "file_info": Tool(
            name="file_info",
            description="Report the line count, byte size and modification time of a file.",
            parameter="path",
            parameter_description="Path relative to the workspace, such as README.md.",
            run=file_info,
        ),
        "json_get": Tool(
            name="json_get",
            description="Read one value out of a JSON file by its dotted path.",
            parameter="query",
            parameter_description="File and path joined by a colon, like run.json:calls.0.name.",
            run=json_get,
        ),
        "today": Tool(
            name="today",
            description="Return the current date and time, optionally offset.",
            parameter="offset",
            parameter_description="Empty for now, or an offset such as '+3 days' or '-2 weeks'.",
            run=today,
        ),
        "run_python": Tool(
            name="run_python",
            description="Run a short Python program and return whatever it prints.",
            parameter="code",
            parameter_description="A complete program that prints its result, like print(2**16).",
            run=run_python,
        ),
    }
