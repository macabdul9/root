"""The HTTP transport, against a fake server so the suite stays offline."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from root.mcp import HTTPClient, MCPTools, MCPUnavailable, ServerSpec, adapt, replies

SESSION = "session-abc"
TOOL = {
    "name": "web_search",
    "description": "Search the web.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "search_queries": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["objective", "search_queries"],
    },
}


class Handler(BaseHTTPRequestHandler):
    """One endpoint speaking just enough streamable HTTP to exercise the client."""

    behaviour = "json"
    seen: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        message = json.loads(body or b"{}")
        Handler.seen.append({"message": message, "headers": dict(self.headers)})
        method = message.get("method")

        if method == "notifications/initialized":
            self.send_response(202)
            self.end_headers()
            return
        if method == "server/discover":
            # What a server older than root's newest protocol actually says.
            self._reply(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
            return
        if method == "initialize":
            self._reply(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"protocolVersion": "2025-06-18"},
                },
                session=SESSION,
            )
            return
        if method == "tools/list":
            self._reply({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": [TOOL]}})
            return
        if method == "tools/call":
            arguments = message["params"]["arguments"]
            self._reply(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(arguments, sort_keys=True)}]
                    },
                }
            )
            return
        self.send_response(400)
        self.end_headers()

    def _reply(self, payload, session=None):
        if Handler.behaviour == "refuse":
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"no"}')
            return
        raw = json.dumps(payload).encode()
        sse = Handler.behaviour == "sse"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if sse else "application/json")
        if session:
            self.send_header("Mcp-Session-Id", session)
        self.end_headers()
        self.wfile.write(f"event: message\ndata: {raw.decode()}\n\n".encode() if sse else raw)


@pytest.fixture
def endpoint():
    Handler.seen = []
    Handler.behaviour = "json"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    server.shutdown()


@pytest.fixture
def client(endpoint):
    started = []

    def make(**overrides):
        spec = ServerSpec(name="remote", url=endpoint, enabled=True, **overrides)
        made = HTTPClient(spec, timeout=10.0, discover_timeout=2.0)
        started.append(made)
        return made

    yield make
    for made in started:
        made.close()


def test_a_remote_server_is_reached_without_spawning_anything(client):
    made = client()
    assert [t["name"] for t in made.list_tools()] == ["web_search"]
    assert made.running


def test_the_handshake_falls_back_when_the_server_is_older_than_root(client):
    """Parallel answers root's newest protocol with an error. Falling back is
    the difference between working and refusing to start."""
    made = client()
    made.start()
    assert made.version == "2025-06-18"
    assert not made.modern


def test_the_session_id_is_echoed_on_every_later_request(client):
    """A server that issues one and never gets it back rejects everything after."""
    made = client()
    made.list_tools()
    later = [s for s in Handler.seen if s["message"].get("method") == "tools/list"]
    assert later and later[0]["headers"].get("Mcp-Session-Id") == SESSION


def test_the_protocol_header_is_withheld_until_a_version_is_settled(client):
    """The header states a settled version, so sending it while settling one is
    what made a real server reject the handshake with HTTP 400."""
    made = client()
    made.list_tools()
    by_method = {s["message"].get("method"): s["headers"] for s in Handler.seen}
    assert "MCP-Protocol-Version" not in by_method["server/discover"]
    assert by_method["tools/list"]["MCP-Protocol-Version"] == "2025-06-18"


def test_a_reply_delivered_as_an_event_stream_is_read(client):
    """The same server answers with JSON or SSE depending on the request."""
    Handler.behaviour = "sse"
    made = client()
    assert [t["name"] for t in made.list_tools()] == ["web_search"]


def test_an_http_error_names_the_server_and_the_status(client):
    Handler.behaviour = "refuse"
    with pytest.raises(MCPUnavailable, match="400"):
        client().list_tools()


def test_an_unreachable_endpoint_is_not_a_crash(client):
    made = HTTPClient(ServerSpec(name="remote", url="http://127.0.0.1:1/mcp", enabled=True))
    with pytest.raises(MCPUnavailable, match="unreachable"):
        made.list_tools()


def test_a_missing_api_key_variable_is_simply_not_sent(client):
    """The key is optional for the free tier, so its absence is not an error."""
    made = client(api_key_env="DEFINITELY_UNSET_KEY_NAME")
    made.list_tools()
    assert "Authorization" not in Handler.seen[-1]["headers"]


def test_the_api_key_is_sent_as_a_bearer_token(client, monkeypatch):
    monkeypatch.setenv("A_TEST_KEY", "secret-value")
    made = client(api_key_env="A_TEST_KEY")
    made.list_tools()
    assert Handler.seen[-1]["headers"]["Authorization"] == "Bearer secret-value"


def test_the_comma_separated_list_arrives_as_an_array(client):
    """The whole point of the adapter: the model writes a string, the server is
    given the array its schema requires."""
    made = client()
    tool, reason = adapt(made, made.list_tools()[0])
    assert tool is not None, reason

    observed = tool.run(objective="find it", search_queries="solar capacity, wind capacity")

    assert '"search_queries": ["solar capacity", "wind capacity"]' in observed


def test_a_url_server_gets_the_http_transport(endpoint):
    registry = MCPTools({"remote": ServerSpec(name="remote", url=endpoint, enabled=True)})
    assert isinstance(registry.clients["remote"], HTTPClient)
    registry.close()


def test_a_notification_answered_with_no_body_carries_nothing():
    class Empty:
        content = b""
        headers = {}

    assert replies(Empty()) == []
