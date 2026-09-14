"""Translate a Dockerfile into an Apptainer definition file.

For hosts that can run containers but not build OCI images. A shared login node
usually has Apptainer and no `/etc/subuid` delegation, which leaves `apptainer
build` from a `.def` working while Docker, rootless podman and Apptainer's own
`dockerfile:` bootstrap all fail.

The ordering is the whole difficulty. Apptainer runs `%files` before `%post`, so
hoisting every COPY above every RUN would quietly reorder a Dockerfile that
copies a checksum file, verifies it, and then copies more. Instead the build
context is staged once and the RUN/COPY sequence is replayed inside `%post` in
its original order, which is faithful for every instruction this handles.

Handles FROM, RUN, COPY, ENV, ARG, WORKDIR and CMD. Multi-stage builds and
COPY --from are refused rather than approximated.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Not under /tmp: Apptainer bind-mounts the host /tmp over the image's during
# %post, so a context staged there is copied in and then shadowed.
STAGING = "/docker-context"

# A line continued with a backslash, and a comment line.
_CONTINUED = re.compile(r"\\\s*$")
_COMMENT = re.compile(r"^\s*#")


class Unsupported(Exception):
    """The Dockerfile uses something this translator will not approximate."""


@dataclass(frozen=True, slots=True)
class Instruction:
    keyword: str
    value: str


def instructions(dockerfile: str) -> list[Instruction]:
    """Logical instructions, with continuations joined and comments dropped."""
    joined: list[str] = []
    buffer = ""
    for raw in dockerfile.splitlines():
        if not buffer and (_COMMENT.match(raw) or not raw.strip()):
            continue
        buffer += raw + "\n"
        if _CONTINUED.search(raw):
            continue
        joined.append(buffer)
        buffer = ""
    if buffer:
        joined.append(buffer)

    parsed = []
    for line in joined:
        keyword, _, value = line.strip().partition(" ")
        parsed.append(Instruction(keyword.upper(), value.strip()))
    return parsed


def flatten(value: str) -> str:
    """One line, for the instructions parsed as words rather than run as shell."""
    return re.sub(r"\s*\\\s*\n\s*", " ", value).strip()


def _resolve(path: str, workdir: str) -> str:
    """A Dockerfile path against the current WORKDIR, as Docker resolves it."""
    if path.startswith("/"):
        return path
    return str(PurePosixPath(workdir or "/") / path)


def _copy_commands(value: str, workdir: str = "") -> list[str]:
    """Shell that reproduces one COPY from the staged context.

    Docker copies the *contents* of a source directory into the destination,
    which `cp -r src dst` only does when dst does not already exist. Copying
    `src/.` is the form that behaves the same either way.
    """
    if value.startswith("--from"):
        raise Unsupported(f"COPY --from needs a multi-stage build: {value!r}")
    parts = shlex.split(value)
    flags = [part for part in parts if part.startswith("--")]
    paths = [part for part in parts if not part.startswith("--")]
    if len(paths) < 2:
        raise Unsupported(f"COPY needs a source and a destination: {value!r}")
    *sources, destination = paths
    trailing = destination.endswith("/")
    destination = _resolve(destination, workdir) + ("/" if trailing else "")

    lines = []
    # A destination ending in / is a directory in Docker, as is a copy of more
    # than one source.
    directory = destination.endswith("/") or len(sources) > 1
    lines.append(
        f"mkdir -p {shlex.quote(destination if directory else str(Path(destination).parent))}"
    )
    for source in sources:
        staged = f"{STAGING}/{source}"
        lines.append(
            f"if [ -d {shlex.quote(staged)} ]; then "
            f"mkdir -p {shlex.quote(destination)}; "
            f"cp -a {shlex.quote(staged + '/.')} {shlex.quote(destination)}; "
            f"else cp -a {shlex.quote(staged)} {shlex.quote(destination)}; fi"
        )
    for flag in flags:
        if flag.startswith("--chown="):
            owner = flag.removeprefix("--chown=")
            lines.append(f"chown -R {shlex.quote(owner)} {shlex.quote(destination)}")
    return lines


def _env_pairs(value: str) -> list[tuple[str, str]]:
    """The KEY=VALUE pairs of one ENV, in either Docker form."""
    if "=" not in value.split()[0]:
        key, _, rest = value.partition(" ")
        return [(key, rest.strip())]
    return [
        (part.split("=", 1)[0], part.split("=", 1)[1]) for part in shlex.split(value) if "=" in part
    ]


def translate(dockerfile: str, context: Path | str) -> str:
    """The Apptainer definition equivalent to this Dockerfile."""
    parsed = instructions(dockerfile)
    froms = [item for item in parsed if item.keyword == "FROM"]
    if not froms:
        raise Unsupported("no FROM instruction")
    if len(froms) > 1:
        raise Unsupported(f"multi-stage build with {len(froms)} FROM instructions")

    base = froms[0].value.split(" AS ")[0].strip()
    post: list[str] = ["set -eux", "export DEBIAN_FRONTEND=noninteractive"]
    environment: list[str] = []
    runscript: list[str] = []
    workdir = ""

    for item in parsed:
        if item.keyword == "FROM":
            continue
        if item.keyword == "RUN":
            body = item.value
            place = f"cd {shlex.quote(workdir)}; " if workdir else ""
            post.append(f"( {place}{body}\n)")
        elif item.keyword == "COPY":
            post.extend(_copy_commands(flatten(item.value), workdir))
        elif item.keyword in ("ENV", "ARG"):
            for key, value in _env_pairs(flatten(item.value)):
                # Exported in %post so later RUNs see it, and again in
                # %environment so the container carries it at run time.
                post.append(f"export {key}={shlex.quote(value)}")
                environment.append(f"export {key}={shlex.quote(value)}")
        elif item.keyword == "WORKDIR":
            workdir = _resolve(flatten(item.value), workdir)
            post.append(f"mkdir -p {shlex.quote(workdir)}")
        elif item.keyword == "CMD":
            runscript.append(item.value.strip("[]").replace('"', "").replace(",", ""))
        else:
            raise Unsupported(f"{item.keyword} is not translated")

    post.append(f"rm -rf {STAGING}")
    if workdir:
        environment.append(f"cd {shlex.quote(workdir)} 2>/dev/null || true")

    sections = [
        "Bootstrap: docker",
        f"From: {base}",
        "",
        "%files",
        f"    {Path(context)} {STAGING}",
        "",
        "%post",
        *(f"    {line}" for command in post for line in command.splitlines()),
    ]
    if environment:
        sections += ["", "%environment", *(f"    {line}" for line in environment)]
    if runscript:
        sections += ["", "%runscript", *(f"    {line}" for line in runscript)]
    return "\n".join(sections) + "\n"
