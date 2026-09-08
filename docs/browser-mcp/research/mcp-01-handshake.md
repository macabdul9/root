## Critical finding first: there are now two eras, and the one you asked about is the legacy one

The **current** protocol version is **`2026-07-28`** (confirmed twice: https://modelcontextprotocol.io/specification/versioning says "The **current** protocol version is 2026-07-28", and `schema/2026-07-28/schema.ts` line 30 declares `export const LATEST_PROTOCOL_VERSION = "2026-07-28";`).

`2026-07-28` **removed the `initialize` handshake entirely.** The spec's own terminology:

- **Modern** = `2026-07-28` and later. Stateless. Version, identity and capabilities travel in `params._meta` on *every single request*. There is no `initialize`, and `notifications/initialized` no longer exists (`schema.ts`: `export type ClientNotification = CancelledNotification;` — the only notification a client may send is `notifications/cancelled`).
- **Legacy** = `2025-11-25` and earlier. `initialize` → response → `notifications/initialized`.

Most servers in the wild are still legacy. Write both paths.

---

## Path A — modern (`2026-07-28`): no handshake

Every request carries this in `params._meta`. In `schema.ts`, `RequestParams` declares `_meta: RequestMetaObject` **non-optional**, so `params` and `params._meta` are mandatory on every request including `tools/list`:

| `_meta` key | Type | Required |
|---|---|---|
| `io.modelcontextprotocol/protocolVersion` | string | **Yes** |
| `io.modelcontextprotocol/clientCapabilities` | object | **Yes** (`{}` is legal and means "no optional capabilities") |
| `io.modelcontextprotocol/clientInfo` | `{name, version}` | No (SHOULD send) |
| `io.modelcontextprotocol/logLevel` | string | No |
| `progressToken` | string/number | No |

A request missing a required field MUST be rejected with `-32602`.

Optional probe (`server/discover`, which servers **MUST** implement). Expects a response:

```json
{"jsonrpc":"2.0","id":"discover-1","method":"server/discover","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"root","version":"0.1.0"},"io.modelcontextprotocol/clientCapabilities":{}}}}
```

Response:

```json
{"jsonrpc":"2.0","id":"discover-1","result":{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{},"resources":{}},"_meta":{"io.modelcontextprotocol/serverInfo":{"name":"ExampleServer","version":"1.0.0"}},"instructions":"This server provides weather and resource utilities.","ttlMs":3600000,"cacheScope":"public"}}
```

`server/discover` is *optional* for a modern-only client — you may fire `tools/list` directly and handle a version error. It is **RECOMMENDED** on stdio even so, because it is also the era probe (below).

## Path B — legacy (`2025-11-25`, `2025-06-18`): `initialize`

Request (expects a response):

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"root","version":"0.1.0"}}}
```

All three of `protocolVersion`, `capabilities`, `clientInfo` are required. `capabilities` may be `{}` — declare only what you implement; declaring `sampling`/`roots`/`elicitation` invites server→client requests you would then have to answer. `clientInfo` requires `name` and `version` (`title`, `description`, `icons`, `websiteUrl` optional).

Response:

```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"logging":{},"prompts":{"listChanged":true},"resources":{"subscribe":true,"listChanged":true},"tools":{"listChanged":true}},"serverInfo":{"name":"ExampleServer","version":"1.0.0"},"instructions":"Optional instructions for the client"}}
```

Version negotiation: if the server supports your version it MUST echo it; otherwise it MUST return another version it supports. **The server can hand back a different version than you asked for** — read `result.protocolVersion`, don't assume. If you can't speak it, disconnect.

Then, mandatory, **no `id`, no response ever comes**:

```json
{"jsonrpc":"2.0","method":"notifications/initialized"}
```

Before the `initialize` response arrives you SHOULD send nothing but `ping`.

## Era detection on stdio (from the 2026-07-28 stdio binding)

Send `server/discover` with your preferred modern version in `_meta`, then:

1. `DiscoverResult` back → modern. Pick from `supportedVersions`.
2. Recognized modern error, e.g. `-32022` → modern, wrong version. Retry with a version from `error.data.supported`. **Do not fall back to `initialize`.**
3. Any other error, or no reply within a timeout → legacy. Fall back to `initialize`.

The fallback **MUST NOT** key on a specific error code — legacy servers answer unknown pre-`initialize` methods with `-32601`, `-32602`, something else, or nothing at all. That "or nothing at all" is why the probe needs a timeout. Cache the era per server process/config.

## Which messages expect a response

| Message | Direction | Response? |
|---|---|---|
| `server/discover` | C→S | Yes |
| `initialize` (legacy) | C→S | Yes |
| `notifications/initialized` (legacy) | C→S | **No** |
| `notifications/cancelled` | C→S | **No** |
| `tools/list` | C→S | Yes |
| `tools/call` | C→S | Yes |
| `notifications/progress`, `notifications/message`, `notifications/tools/list_changed` | S→C | **No** (and they arrive interleaved on the same stdout stream) |

## Shutdown

Close the child's stdin, wait, then SIGTERM, then SIGKILL. Servers SHOULD exit on EOF of stdin; that's the only portable graceful signal.