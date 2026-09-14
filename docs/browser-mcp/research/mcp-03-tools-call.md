## `tools/call`

**Request** (modern; drop the `_meta` block for legacy):

```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_weather","arguments":{"location":"New York"},"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"root","version":"0.1.0"},"io.modelcontextprotocol/clientCapabilities":{}}}}
```

`arguments` is optional in the schema (`arguments?: { [key: string]: unknown }`), but send `{}` rather than omitting it — some servers are stricter than the schema.

**Response:**

```json
{"jsonrpc":"2.0","id":3,"result":{"resultType":"complete","content":[{"type":"text","text":"Current weather in New York:\nTemperature: 72°F\nConditions: Partly cloudy"}],"isError":false}}
```

`CallToolResult` per `schema.ts`: `content: ContentBlock[]` (required), `structuredContent?: unknown`, `isError?: boolean` (absent means `false`).

### Content block types

```json
{"type":"text","text":"Tool result text"}
{"type":"image","data":"<base64>","mimeType":"image/png"}
{"type":"audio","data":"<base64>","mimeType":"audio/wav"}
{"type":"resource_link","uri":"file:///project/src/main.rs","name":"main.rs","description":"Primary application entry point","mimeType":"text/x-rust"}
{"type":"resource","resource":{"uri":"file:///project/src/main.rs","mimeType":"text/x-rust","text":"fn main() {}"}}
```

All five may carry an `annotations` object. For a 350M-parameter model you want one flat string: keep `type == "text"` blocks, join on `"\n"`, and render the rest as a short placeholder (`[image image/png]`) rather than pasting base64 into the context window. Then apply your `MAX_TOOL_OUTPUT_CHARS = 800` truncation — a single `tools/call` can legitimately return megabytes.

### `structuredContent`

Any JSON value conforming to the tool's `outputSchema`. When a tool returns it, the server SHOULD *also* put the serialized JSON in a text block, so a text-only client loses nothing:

```json
{"jsonrpc":"2.0","id":5,"result":{"resultType":"complete","content":[{"type":"text","text":"{\"temperature\": 22.5, \"conditions\": \"Partly cloudy\", \"humidity\": 65}"}],"structuredContent":{"temperature":22.5,"conditions":"Partly cloudy","humidity":65}}}
```

### `resultType` — new in 2026-07-28, and it can mean "not done"

Every modern result carries `resultType`. **Absent means `"complete"`** (mandated backward-compat rule for legacy servers). Any value your client does not recognize MUST be treated as invalid.

`"input_required"` means the server needs client-side input (elicitation/sampling/roots) before it can finish — Multi Round-Trip Requests:

```json
{"jsonrpc":"2.0","id":3,"result":{"resultType":"input_required","inputRequests":{"github_login":{"method":"elicitation/create","params":{"mode":"form","message":"Please provide your GitHub username","requestedSchema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}},"requestState":"eyJsb2NhdGlvbiI6Ik5ldyBZb3JrIn0..."}}
```

You retry as a **new request with a different `id`**, echoing `requestState`:

```json
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"get_weather","arguments":{"location":"New York"},"inputResponses":{"github_login":{"action":"accept","content":{"name":"octocat"}}},"requestState":"eyJsb2NhdGlvbiI6Ik5ldyBZb3JrIn0...","_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"root","version":"0.1.0"},"io.modelcontextprotocol/clientCapabilities":{}}}}
```

Simplest correct behaviour for a minimal client: declare `clientCapabilities: {}`, which means servers MUST NOT require elicitation of you (they must return `-32021` instead). If `input_required` still arrives, surface it as a tool error string rather than pretending the call succeeded.

Note the architectural consequence of MRTR on stdio: **the server MUST NOT write JSON-RPC *requests* to stdout** in 2026-07-28. Server→client interaction is carried inside `InputRequiredResult` replies. Under legacy versions a server *can* send you real requests (`sampling/createMessage`, `roots/list`, `elicitation/create`) — but only if you declared those capabilities. Declaring `{}` is what keeps your read loop from needing a request handler.