## Framing: newline-delimited, no headers — and here is exactly what the spec guarantees

Normative bullets from the `2026-07-28` stdio binding (`2025-06-18` is word-for-word the same on the first four):

- "Each message is a single JSON-RPC request, notification, or response."
- "Messages are delimited by newlines, and **MUST NOT** contain embedded newlines."
- "The server **MUST NOT** write anything to its `stdout` that is not a valid MCP message."
- "The client **MUST NOT** write anything to the server's `stdin` that is not a valid MCP message."
- "The server **MAY** write UTF-8 strings to `stderr` for any logging purposes" and the client "**SHOULD NOT** assume `stderr` output indicates error conditions."
- All JSON-RPC messages MUST be UTF-8.

**There are no `Content-Length` headers.** That is LSP (and the old `2024-11-05` HTTP+SSE binding is a separate thing again). If you have written an LSP client before, this is the single highest-value thing to unlearn — MCP stdio is one JSON object per line, full stop.

## The failure modes that will actually bite you

**1. Pretty-printed JSON silently corrupts the stream.** `json.dumps(msg, indent=2)` emits embedded newlines, which the spec forbids. Compact separators, one `\n` appended, then flush:

```python
proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
proc.stdin.flush()
```

Note `json.dumps` escapes literal newlines *inside string values* as `\n` (two chars), so message content is safe automatically. Only your own formatting can break the frame.

**2. Not flushing, and buffering on both sides.** Launch with `bufsize=1, text=True, encoding="utf-8"` and flush after every write. Without the flush your `initialize` sits in a 8 KiB buffer and the client hangs waiting for a reply to a message that was never sent. For Python child servers, pass `PYTHONUNBUFFERED=1` in `env`.

**3. Waiting for a response to a notification.** `notifications/initialized` and `notifications/cancelled` have no `id` and get no reply. A read loop shaped `write(); return readline()` deadlocks on the very first notification. Structure it as: writer, plus a reader that dispatches by `id` presence.

**4. `readline()` on the wrong response.** The server interleaves notifications (`notifications/progress`, `notifications/message`, `notifications/tools/list_changed`) with responses on the same stdout. A client that assumes the next line is its answer will parse a progress notification as a `tools/call` result. Loop until you read a line whose `id` matches; buffer or drop everything else. Even a strictly serial client needs this loop.

**5. Not draining stderr → deadlock.** If you use `stderr=subprocess.PIPE` and never read it, a chatty server fills the ~64 KiB pipe buffer, blocks on write, stops answering, and your client hangs on a read that will never complete. Either `stderr=subprocess.DEVNULL`, redirect to a file, or drain it on a daemon thread. And per spec, stderr output is *not* an error signal — do not fail on it.

**6. Servers that pollute stdout anyway.** The MUST NOT is routinely violated: startup banners, a stray `print()`, a dependency's warning. Your reader must skip lines that don't parse as JSON or lack `"jsonrpc"` (log them at debug) instead of raising. Also skip empty lines. Conversely, be scrupulous on your own side: never `print()` to the child's stdin.

**7. `readline()` returning `""` means the process died.** Not "no data yet". Check `proc.poll()`, surface the exit code plus captured stderr, and don't retry into an infinite loop. Blank-line-vs-EOF confusion produces a busy spin.

**8. EOF is the shutdown handshake.** Close stdin, `proc.wait(timeout=...)`, then `terminate()`, then `kill()`. Don't SIGKILL first; servers with open handles need the graceful path.

**9. `_meta` on *every* modern request, including `tools/list`.** The tools docs page omits `_meta` from its examples and flags the omission in a note; `schema.ts` declares `RequestParams._meta` as non-optional. Build the `_meta` block once in a helper that every outbound request goes through, or you will ship a client that works against `server/discover` and gets `-32602` on `tools/list`.

**10. Assuming `2026-07-28` because it's current.** Any server built on a pre-mid-2026 SDK is legacy and will reject modern requests, possibly by *silently processing them under legacy semantics* — the spec calls out that some legacy servers don't validate that a request arrives after `initialize`, so a bare `tools/call` may appear to work while skipping negotiation. Probe with `server/discover` and fall back; and remember the "no response at all" outcome, so the probe needs its own timeout.

**11. Windows encoding.** Set `encoding="utf-8"` explicitly on the pipes; the default is the ANSI code page and will mangle non-ASCII tool output.

**12. Reusing request ids.** MUST be unique among outstanding requests and MUST NOT be `null`. A monotonic `itertools.count(1)` is fine; a per-call `id: 1` is not.

## Sources

- https://modelcontextprotocol.io/specification/versioning
- https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning
- https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio
- https://modelcontextprotocol.io/specification/2026-07-28/basic/index
- https://modelcontextprotocol.io/specification/2026-07-28/server/discover
- https://modelcontextprotocol.io/specification/2026-07-28/server/tools
- https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle
- https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle
- https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
- https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/schema/2026-07-28/schema.ts

The schema is saved locally at `/private/tmp/claude-501/-Users-awaheed-Research-AIAgents/249d0cf8-e005-408b-8151-8abc211c8b61/scratchpad/schema.ts` (96 KB) — it is the normative source of truth and worth grepping while implementing.