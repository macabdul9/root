from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from typing import IO

FRAME_SECONDS = 0.25
FRAMES = ("   ", ".  ", ".. ", "...")


@contextmanager
def working(message: str, stream: IO[str] | None = None):
    """Show a message with moving dots while something slow happens.

    Written to stderr, so a piped or redirected run keeps a clean stdout, and
    animated only for a terminal: anywhere else the message is printed once and
    left alone rather than filling a log with carriage returns.
    """
    stream = stream or sys.stderr
    if not stream.isatty():
        print(f"{message}...", file=stream, flush=True)
        yield
        return

    done = threading.Event()

    def spin() -> None:
        for frame in _cycle():
            if done.wait(FRAME_SECONDS):
                break
            print(f"\r{message}{frame}", end="", file=stream, flush=True)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=1)
        print("\r" + " " * (len(message) + len(FRAMES[-1])) + "\r", end="", file=stream, flush=True)


def _cycle():
    while True:
        yield from FRAMES
