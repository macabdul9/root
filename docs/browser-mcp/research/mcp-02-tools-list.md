## `tools/list`

**Modern (`2026-07-28`).** The docs page shows the request without `_meta` and says explicitly it omits it "for brevity" — the schema requires it. Send:

```json
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"root","version":"0.1.0"},"io.modelcontextprotocol/clientCapabilities":{}}}}
```

Add `"cursor": "<opaque>"` alongside `_meta` in `params` to page.

**Legacy.** `params` is optional entirely:

```json
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
```

**Response** (modern; legacy is identical minus `resultType`/`ttlMs`/`cacheScope`):

```json
{"jsonrpc":"2.0","id":2,"result":{"resultType":"complete","tools":[{"name":"get_weather","title":"Weather Information Provider","description":"Get current weather information for a location","inputSchema":{"type":"object","properties":{"location":{"type":"string","description":"City name or zip code"}},"required":["location"]},"icons":[{"src":"https://example.com/weather-icon.png","mimeType":"image/png","sizes":["48x48"]}]}],"nextCursor":"next-page-cursor","ttlMs":300000,"cacheScope":"public"}}
```

Pagination: loop while `result.nextCursor` is present, feeding it back as `params.cursor`. **`nextCursor` absent means done** — do not treat an empty `tools` array as the terminator, and do not treat a present-but-empty `nextCursor` as meaningful; test for key presence.

### Tool object fields you'll actually consume

- `name` — required. 1–128 chars, case-sensitive, `[A-Za-z0-9_.-]`, unique per server. Not unique *across* servers: if you multiplex, prefix them. Note `.` and `-` are legal, so these names are not always valid Python identifiers and won't always survive a naive mapping into your `Tool` dataclass namespace.
- `title` — optional display name.
- `description` — optional in the schema, though every real server sends one.
- `inputSchema` — required, MUST be a JSON Schema object, never `null`. Defaults to JSON Schema 2020-12 when `$schema` is absent. A no-arg tool is `{"type":"object","additionalProperties":false}` or `{"type":"object"}`.
- `outputSchema` — optional; if present the server MUST return conforming `structuredContent`.
- `annotations` — optional, **explicitly untrusted** per the spec.

### Relevance to `root`'s two-string-argument convention

`inputSchema` is arbitrary JSON Schema — nested objects, arrays, enums, `anyOf`. Your `Parameter(name, description, freeform=False)` model cannot represent that. Decide the projection deliberately at adapter time: read `inputSchema["properties"]` and `inputSchema.get("required", [])`, and either skip tools whose required-parameter count exceeds two or whose required params aren't string/number/boolean, or expose one `freeform=True` parameter carrying raw JSON. Silently dropping parameters produces calls the server rejects with `-32602`.

Do not dereference network `$ref`s while validating (spec: MUST NOT auto-dereference; opt-in only, with an allowlist).