## Two distinct failure channels — do not collapse them

**1. Protocol errors** — JSON-RPC `error` object. Unknown tool, malformed request, server fault:

```json
{"jsonrpc":"2.0","id":3,"error":{"code":-32602,"message":"Unknown tool: invalid_tool_name"}}
```

**2. Tool execution errors** — a *successful* JSON-RPC response with `isError: true`:

```json
{"jsonrpc":"2.0","id":4,"result":{"resultType":"complete","content":[{"type":"text","text":"Invalid departure date: must be in the future. Current date is 08/08/2025."}],"isError":true}}
```

The distinction is behavioural, and the spec is explicit about it: tool execution errors SHOULD be fed back to the model so it can self-correct and retry with different arguments; protocol errors MAY be, but rarely help. A client that raises a Python exception on `isError: true` destroys the retry loop that is the whole point of the split. Feed the `content` text back as the tool result, `isError` flag and all.

## Error codes

Standard JSON-RPC: `-32700` parse error, `-32600` invalid request, `-32601` method not found, `-32602` invalid params, `-32603` internal error.

MCP partitions the implementation-defined range:
- `-32000`..`-32019` — **legacy, do not allocate, assume no meaning** (except `-32002`).
- `-32020`..`-32099` — reserved for the MCP spec.

Defined in `2026-07-28`:

| Code | Name | When |
|---|---|---|
| `-32020` | `HeaderMismatch` | HTTP only — headers disagree with body |
| `-32021` | `MissingRequiredClientCapability` | You didn't declare a capability the request needs; `data.requiredCapabilities` lists them |
| `-32022` | `UnsupportedProtocolVersion` | Your `_meta` version isn't supported |

Retired but still received from older servers: `-32002` (resource not found; replaced by `-32602`) — clients SHOULD still accept it. `-32042` (URL elicitation, `2025-11-25` only).

`UnsupportedProtocolVersionError` has a **guaranteed `data` shape** — this is the one error worth parsing structurally:

```json
{"jsonrpc":"2.0","id":1,"error":{"code":-32022,"message":"Unsupported protocol version","data":{"supported":["2026-07-28","2025-11-25"],"requested":"1900-01-01"}}}
```

Select a mutually supported version from `data.supported` and retry the same request with a new `id`.

The legacy equivalent looks similar but uses `-32602` and arrives in reply to `initialize`:

```json
{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"Unsupported protocol version","data":{"supported":["2024-11-05"],"requested":"1.0.0"}}}
```

## Error-response shape rules

- `error` MUST have integer `code` and string `message`; `data` is optional and any type.
- `id` matches the request — **except** when the request was so malformed the id couldn't be read, in which case `id` may be absent or `null`. Your dispatch table needs a branch for a response you cannot correlate; log it, don't crash.
- Request `id` MUST NOT be `null` (stricter than base JSON-RPC) and MUST NOT reuse an id that is still outstanding.

## Timeouts and cancellation

Set a per-request timeout. On expiry, send `notifications/cancelled` referencing the id and stop waiting:

```json
{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":3,"reason":"timeout"}}
```

Then **keep the id in a discard set** — the server may already have the response in flight, and a late reply for an abandoned id must not be matched to a later request. You may reset the timeout on `notifications/progress` for that id, but still enforce a hard ceiling.