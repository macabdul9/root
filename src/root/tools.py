from __future__ import annotations

import ast
import csv
import json
import operator
import os
import re
import sqlite3
import subprocess
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

MAX_TOOL_OUTPUT_CHARS = 800
MAX_LISTED_FILES = 40
MAX_GREP_MATCHES = 30
MAX_PDF_PAGES = 20
GIT_SUBCOMMANDS = frozenset({"status", "log", "diff", "branch", "show", "remote", "blame"})
GIT_TIMEOUT_SECONDS = 15
MAX_GREP_FILE_BYTES = 1_000_000
PYTHON_TIMEOUT_SECONDS = 10

SKIPPED_DIRS = frozenset(
    {".git", ".venv", "venv", "__pycache__", "runs", ".pytest_cache", ".ruff_cache", "node_modules"}
)

_OFFSET = re.compile(r"^([+-]?\d+)\s*(second|minute|hour|day|week)s?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Parameter:
    """One string argument a tool takes.

    `freeform` marks a value with no shape worth constraining - a program, a
    file's contents. Forcing those through a grammar measurably degrades what
    the model writes: constrained, the python agent scored 11/20 against 13
    unconstrained, because the quoting rules crowd out the code.
    """

    name: str
    description: str
    freeform: bool = False


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool the model calls with string arguments.

    Nearly every tool takes one, which is what a sub-billion-parameter model
    gets right most reliably. Writing a file needs two, so the shape allows it
    without encouraging it.
    """

    name: str
    description: str
    parameters: tuple[Parameter, ...]
    run: Callable[..., str]

    @property
    def parameter(self) -> str:
        return self.parameters[0].name

    @property
    def constrainable(self) -> bool:
        return not any(parameter.freeform for parameter in self.parameters)

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        parameter.name: {
                            "type": "string",
                            "description": parameter.description,
                            **({"x-freeform": True} if parameter.freeform else {}),
                        }
                        for parameter in self.parameters
                    },
                    "required": [parameter.name for parameter in self.parameters],
                },
            },
        }

    def bind(self, arguments: list[tuple[str | None, str]]) -> dict[str, str]:
        """Match parsed arguments to parameters.

        Named arguments go where they are named; unnamed ones fill what is left,
        in order. A model that writes `write_file('a.txt', 'hi')` and one that
        writes `write_file(content='hi', path='a.txt')` both land correctly.
        """
        known = {parameter.name for parameter in self.parameters}
        bound = {name: value for name, value in arguments if name in known}
        unfilled = [p.name for p in self.parameters if p.name not in bound]
        spare = [value for name, value in arguments if name not in known]
        bound.update(zip(unfilled, spare, strict=False))
        missing = known - set(bound)
        if missing:
            raise ValueError(f"{self.name} needs {sorted(missing)}")
        return bound


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


def balance_parentheses(code: str) -> str:
    """Drop trailing ")" that the model added past the end of its own call.

    `sorted([2, 3, 41]))` is a slip it makes repeatedly and cannot recover from:
    told about the SyntaxError it emits the same program again. Only applied
    when dropping them turns unparseable code into parseable code.
    """
    if _parses(code):
        return code
    trimmed = code.rstrip()
    while trimmed.endswith(")"):
        trimmed = trimmed[:-1].rstrip()
        if _parses(trimmed):
            return trimmed
    return code


def _parses(code: str) -> bool:
    """Whether model-written code compiles, quietly.

    ast.parse warns on things like an invalid escape, and that warning goes to
    the terminal from inside a tool the user did not know was parsing anything.
    Whether the code is valid is the only thing wanted here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ast.parse(code)
        except SyntaxError:
            return False
    return True


def _column_name(name: str, index: int) -> str:
    """A usable SQL identifier for a CSV header cell."""
    cleaned = re.sub(r"\W+", "_", name.strip()).strip("_")
    return cleaned or f"column{index + 1}"


def ensure_print(code: str) -> str:
    """Wrap a bare expression in print().

    Small models write `sum(range(1, 21))` with no print, get nothing back, and
    then invent a number rather than retrying.
    """
    if "print(" in code:
        return code
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

    def read_file(path: str) -> str:
        resolved = _resolve_inside(workspace, path)
        if not resolved.is_file():
            raise FileNotFoundError(f"no such file: {path.strip()}")
        text = resolved.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            return text[:MAX_TOOL_OUTPUT_CHARS] + "\n[truncated]"
        return text

    def csv_query(path: str, query: str) -> str:
        """Run one SQL statement over a CSV file, which is loaded as `data`.

        The python agent's failures are arithmetic inside code the model wrote:
        43 words where there were 9, 13 s's where there were 7. SQL takes the
        model out of the counting the way `calculator` already does, and SQLite
        does the arithmetic exactly.
        """
        resolved = _resolve_inside(workspace, path)
        if not resolved.is_file():
            raise FileNotFoundError(f"no such file: {path.strip()}")
        statement = query.strip().rstrip(";")
        if not statement:
            raise ValueError("expected a SQL query, such as SELECT count(*) FROM data")
        if not statement.lower().startswith(("select", "with")):
            raise ValueError("only SELECT queries are allowed")

        with resolved.open(newline="", encoding="utf-8", errors="replace") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return f"{path.strip()} is empty"
        header, body = rows[0], rows[1:]
        columns = [_column_name(name, index) for index, name in enumerate(header)]

        connection = sqlite3.connect(":memory:")
        try:
            placeholders = ", ".join("?" for _ in columns)
            quoted = ", ".join(f'"{name}"' for name in columns)
            connection.execute(f"CREATE TABLE data ({quoted})")
            connection.executemany(f"INSERT INTO data VALUES ({placeholders})", body)
            cursor = connection.execute(statement)
            names = [description[0] for description in cursor.description or []]
            found = cursor.fetchmany(MAX_LISTED_FILES)
        except sqlite3.Error as exc:
            raise ValueError(f"{exc}; the table is called data with columns {columns}") from None
        finally:
            connection.close()

        if not found:
            return "no rows"
        lines = [", ".join(names)] + [
            ", ".join("" if v is None else str(v) for v in r) for r in found
        ]
        return "\n".join(lines)[:MAX_TOOL_OUTPUT_CHARS]

    def read_pdf(path: str) -> str:
        """Extract the text layer of a PDF.

        A scanned PDF has no text layer and returns nothing useful; this does no
        OCR, and says so rather than returning an empty string.
        """
        from pypdf import PdfReader

        resolved = _resolve_inside(workspace, path)
        if not resolved.is_file():
            raise FileNotFoundError(f"no such file: {path.strip()}")
        pages = PdfReader(resolved).pages
        text = "\n".join(page.extract_text() or "" for page in pages[:MAX_PDF_PAGES]).strip()
        if not text:
            return f"{len(pages)} pages with no text layer; this reads text, it does not do OCR"
        note = ""
        if len(pages) > MAX_PDF_PAGES:
            note = f"\n[first {MAX_PDF_PAGES} of {len(pages)} pages]"
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            return text[:MAX_TOOL_OUTPUT_CHARS] + "\n[truncated]" + note
        return text + note

    def read_lines(path: str, lines: str) -> str:
        """Read one span of a file, for when the whole thing does not fit.

        `read_file` truncates at 800 characters, which is no use past the top of
        a long file; this takes `40-80` or a single line number.
        """
        resolved = _resolve_inside(workspace, path)
        if not resolved.is_file():
            raise FileNotFoundError(f"no such file: {path.strip()}")
        span = lines.strip().replace(" ", "")
        try:
            first, _, last = span.partition("-")
            start = int(first)
            end = int(last) if last else start
        except ValueError:
            raise ValueError(f"expected a line range like 40-80, got {lines.strip()!r}") from None
        if start < 1 or end < start:
            raise ValueError(f"line range must count from 1 and go forwards, got {span!r}")

        numbered = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        chosen = numbered[start - 1 : end]
        if not chosen:
            return f"{path.strip()} has {len(numbered)} lines; {span} is past the end"
        body = "\n".join(f"{number}: {line}" for number, line in enumerate(chosen, start=start))
        return body[:MAX_TOOL_OUTPUT_CHARS]

    def append_file(path: str, content: str) -> str:
        """Add to the end of a file, creating it when missing."""
        target = _resolve_inside(workspace, path)
        if target.is_dir():
            raise IsADirectoryError(f"{path.strip()} is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_text(encoding="utf-8") if target.is_file() else ""
        separator = "" if not existing or existing.endswith("\n") else "\n"
        target.write_text(existing + separator + content, encoding="utf-8")
        return f"appended {len(content)} characters to {target.relative_to(workspace)}"

    def move_file(source: str, destination: str) -> str:
        """Rename or move a file, without overwriting anything."""
        origin = _resolve_inside(workspace, source)
        target = _resolve_inside(workspace, destination)
        if not origin.exists():
            raise FileNotFoundError(f"no such file: {source.strip()}")
        if target.exists():
            raise FileExistsError(f"{destination.strip()} already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        origin.rename(target)
        return f"moved {origin.relative_to(workspace)} to {target.relative_to(workspace)}"

    def delete_file(path: str) -> str:
        """Delete one file.

        Files only: a directory can hold work the model never saw, and removing
        a tree on a small model's say-so is not a risk worth taking.
        """
        target = _resolve_inside(workspace, path)
        if target.is_dir():
            raise IsADirectoryError(f"{path.strip()} is a directory; this deletes files only")
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path.strip()}")
        target.unlink()
        return f"deleted {target.relative_to(workspace)}"

    def git(command: str) -> str:
        """Run one read-only git command in the workspace.

        Allowlisted rather than a general shell: the model gets to inspect the
        repository, not to rewrite it.
        """
        parts = command.strip().removeprefix("git ").split()
        if not parts:
            raise ValueError(f"expected a subcommand, one of {sorted(GIT_SUBCOMMANDS)}")
        if parts[0] not in GIT_SUBCOMMANDS:
            raise ValueError(f"{parts[0]!r} is not allowed; try {sorted(GIT_SUBCOMMANDS)}")
        completed = subprocess.run(
            ["git", *parts],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
        output = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
            return f"error: git {parts[0]} failed\n{output[:MAX_TOOL_OUTPUT_CHARS]}"
        return output[:MAX_TOOL_OUTPUT_CHARS] or f"git {parts[0]} printed nothing"

    def write_file(path: str, content: str) -> str:
        """Write a file, creating parent directories as needed.

        Overwrites without asking. The workspace check is the only guard, so
        point the session at a directory you are willing to have rewritten.
        """
        target = _resolve_inside(workspace, path)
        if target.is_dir():
            raise IsADirectoryError(f"{path.strip()} is a directory")
        existed = target.is_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        verb = "overwrote" if existed else "wrote"
        return f"{verb} {target.relative_to(workspace)}, {len(content)} characters"

    def make_directory(path: str) -> str:
        target = _resolve_inside(workspace, path)
        if target.is_file():
            raise NotADirectoryError(f"{path.strip()} is a file")
        existed = target.is_dir()
        target.mkdir(parents=True, exist_ok=True)
        state = "already exists" if existed else "created"
        return f"{target.relative_to(workspace)} {state}"

    def list_files(pattern: str) -> str:
        pattern = pattern.strip().strip("'\"") or "*"
        if ".." in Path(pattern).parts:
            raise ValueError(f"pattern is outside the workspace: {pattern}")
        # Models pass a directory name far more often than a glob that matches
        # one, and `Path.glob(".")` raises rather than listing anything.
        if pattern in {".", "./"} or (workspace / pattern).is_dir():
            pattern = "*" if pattern in {".", "./"} else f"{pattern.rstrip('/')}/*"
        try:
            found = list(workspace.glob(pattern))
        except (IndexError, ValueError, NotImplementedError):
            raise ValueError(f"not a usable glob pattern: {pattern}") from None
        matches = sorted(str(p.relative_to(workspace)) for p in found if p.is_file())
        if not matches:
            return f"no files match {pattern}"
        listed = matches[:MAX_LISTED_FILES]
        if len(matches) > len(listed):
            listed.append(f"[{len(matches) - len(listed)} more]")
        return "\n".join(listed)

    def grep(pattern: str) -> str:
        pattern = pattern.strip()
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

    def file_info(path: str) -> str:
        resolved = _resolve_inside(workspace, path)
        if not resolved.is_file():
            raise FileNotFoundError(f"no such file: {path.strip()}")
        stat = resolved.stat()
        lines = resolved.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        return (
            f"{resolved.relative_to(workspace)}: {lines} lines, "
            f"{stat.st_size} bytes, modified {modified}"
        )

    def json_get(query: str) -> str:
        if ":" not in query:
            raise ValueError("expected 'file.json:dotted.path', for example config.json:model.name")
        raw_path, dotted = query.split(":", 1)
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
        code = ensure_print(balance_parentheses(code.strip()))
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
            parameters=(
                Parameter(
                    "expression",
                    "Arithmetic over numbers, such as 24 * 7 - 13.",
                ),
            ),
            run=calculate,
        ),
        "run_python": Tool(
            name="run_python",
            description="Run a short Python program and return whatever it prints.",
            parameters=(
                Parameter(
                    "code",
                    "A complete program that prints its result, like print(2**16).",
                    freeform=True,
                ),
            ),
            run=run_python,
        ),
        "csv_query": Tool(
            name="csv_query",
            description="Run a SQL SELECT over a CSV file. The table is called data.",
            parameters=(
                Parameter(
                    "path",
                    "Path relative to the workspace, such as data/records.csv.",
                ),
                Parameter(
                    "query",
                    "A SELECT statement over the table data, such as "
                    "SELECT count(*) FROM data WHERE score > 50.",
                ),
            ),
            run=csv_query,
        ),
        "read_pdf": Tool(
            name="read_pdf",
            description="Extract the text of a PDF in the workspace.",
            parameters=(
                Parameter(
                    "path",
                    "Path relative to the workspace, such as report.pdf.",
                ),
            ),
            run=read_pdf,
        ),
        "read_lines": Tool(
            name="read_lines",
            description="Read a numbered span of lines from a file.",
            parameters=(
                Parameter(
                    "path",
                    "Path relative to the workspace, such as README.md or src/app.py.",
                ),
                Parameter(
                    "lines",
                    "A line range like 40-80, or a single line number.",
                ),
            ),
            run=read_lines,
        ),
        "read_file": Tool(
            name="read_file",
            description="Read a text file from the workspace and return its contents.",
            parameters=(
                Parameter(
                    "path",
                    "Path relative to the workspace, such as README.md or src/app.py.",
                ),
            ),
            run=read_file,
        ),
        "write_file": Tool(
            name="write_file",
            description="Create or overwrite a text file in the workspace.",
            parameters=(
                Parameter(
                    "path",
                    "Path relative to the workspace, such as notes/todo.md.",
                ),
                Parameter(
                    "content",
                    "The complete text to write. It replaces the file.",
                ),
            ),
            run=write_file,
        ),
        "append_file": Tool(
            name="append_file",
            description="Add text to the end of a workspace file, creating it if missing.",
            parameters=(
                Parameter(
                    "path",
                    "Path relative to the workspace, such as notes/todo.md.",
                ),
                Parameter(
                    "content",
                    "The text to add. A newline is inserted first when needed.",
                ),
            ),
            run=append_file,
        ),
        "move_file": Tool(
            name="move_file",
            description="Rename or move a workspace file. Refuses to overwrite.",
            parameters=(
                Parameter("source", "The existing path."),
                Parameter("destination", "Where it should end up."),
            ),
            run=move_file,
        ),
        "delete_file": Tool(
            name="delete_file",
            description="Delete one file from the workspace. Files only, not directories.",
            parameters=(
                Parameter(
                    "path",
                    "Path relative to the workspace, such as scratch.txt.",
                ),
            ),
            run=delete_file,
        ),
        "git": Tool(
            name="git",
            description="Run a read-only git command in the workspace and return its output.",
            parameters=(
                Parameter(
                    "command",
                    "A subcommand such as status, log --oneline -5, or diff --stat.",
                ),
            ),
            run=git,
        ),
        "make_directory": Tool(
            name="make_directory",
            description="Create a directory in the workspace, including parents.",
            parameters=(
                Parameter(
                    "path",
                    "Path relative to the workspace, such as src/models.",
                ),
            ),
            run=make_directory,
        ),
        "list_files": Tool(
            name="list_files",
            description="List workspace files matching a glob pattern.",
            parameters=(
                Parameter(
                    "pattern",
                    "Glob pattern such as *.py or src/**/*.md.",
                ),
            ),
            run=list_files,
        ),
        "grep": Tool(
            name="grep",
            description="Search every workspace file for a pattern and return matching lines.",
            parameters=(
                Parameter(
                    "pattern",
                    "Text or regular expression, such as force_first_call.",
                ),
            ),
            run=grep,
        ),
        "file_info": Tool(
            name="file_info",
            description="Report the line count, byte size and modification time of a file.",
            parameters=(
                Parameter(
                    "path",
                    "Path relative to the workspace, such as README.md.",
                ),
            ),
            run=file_info,
        ),
        "json_get": Tool(
            name="json_get",
            description="Read one value out of a JSON file by its dotted path.",
            parameters=(
                Parameter(
                    "query",
                    "File and path joined by a colon, like run.json:calls.0.name.",
                ),
            ),
            run=json_get,
        ),
        "today": Tool(
            name="today",
            description="Return the current date and time, optionally offset.",
            parameters=(
                Parameter(
                    "offset",
                    "Empty for now, or an offset such as '+3 days' or '-2 weeks'.",
                ),
            ),
            run=today,
        ),
    }
