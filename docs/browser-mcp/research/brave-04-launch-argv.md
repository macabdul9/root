The project publishes its own image to Docker Hub. From .github/workflows/build-release.yml, the tag list is built from DOCKER_REGISTRY = ${{ github.repository }}, i.e. the repo path, so the image is docker.io/brave/brave-search-mcp-server, tagged v<version> (plus full and short commit SHAs). I confirmed against the Docker Hub tags API: 251 tags, v2.1.3 pushed 2026-08-20. There is deliberately NO `latest` tag ("per production best practice, remove 'latest' tag"), so you must pin. The same build is mirrored to an AWS Marketplace ECR repo as brave/brave-search-mcp, which needs Marketplace auth -- use Docker Hub. (The README's docker.io/mcp/brave-search is Docker's own MCP-catalog rebuild, not Brave's; server.json for v2.1.3 lists only the npm package, no OCI entry.)

The Dockerfile's ENTRYPOINT is ["node","dist/index.js"] and src/index.js defaults to the stdio transport, so anything after the image name is appended as CLI args to the server.

Exact argv:

  ["docker","run","-i","--rm","-e","BRAVE_API_KEY","brave/brave-search-mcp-server:v2.1.3"]

With your whitelist and the transport made explicit:

  ["docker","run","-i","--rm",
   "-e","BRAVE_API_KEY",
   "brave/brave-search-mcp-server:v2.1.3",
   "--transport","stdio",
   "--enabled-tools","brave_web_search","brave_news_search","brave_video_search","brave_image_search"]

Key passing: `-e BRAVE_API_KEY` with NO `=value` tells docker to copy the variable from the parent process's environment. That is the form to use -- the secret never lands in argv, so it stays out of `ps`, out of any subprocess log, and out of the AgentSpec you would otherwise commit. The variable must be exported in whatever process spawns docker (root's terminal harness, or the shell running pytest). If you prefer to inline it, `-e","BRAVE_API_KEY=sk-...` also works, and the server equally accepts a `--brave-api-key <string>` flag after the image name, but both put the key in the process table.

Mechanics that matter:
  -i is REQUIRED. stdio transport needs stdin held open; drop it and the server sees EOF and exits immediately.
  Do NOT pass -t. A TTY corrupts the newline-delimited JSON-RPC framing.
  --rm keeps a container from accumulating per agent run.
  No -p and no BRAVE_MCP_HOST concern: the image sets ENV BRAVE_MCP_HOST=0.0.0.0, but that only applies to the http transport, which you are not using.
  Image runs as USER node, non-root already.

Hardened variant, matching the flags Brave themselves ship as usage instructions:

  ["docker","run","-i","--rm","--cap-drop","all","--read-only",
   "-e","BRAVE_API_KEY",
   "brave/brave-search-mcp-server:v2.1.3","--transport","stdio"]

If you would rather keep the key in a file than the environment (fits a configs/ layout, and takes precedence over BRAVE_API_KEY):

  ["docker","run","-i","--rm",
   "-v","/absolute/path/to/brave.key:/run/secrets/brave.key:ro",
   "-e","BRAVE_API_KEY_FILE=/run/secrets/brave.key",
   "brave/brave-search-mcp-server:v2.1.3","--transport","stdio"]

First run needs network to pull; after `docker pull brave/brave-search-mcp-server:v2.1.3` the launch is offline apart from the calls to api.search.brave.com. Startup fails fast with "Error: A Brave API key is required ..." on stderr and exit 1 if no key reaches it, which is the signal your tests should assert on rather than requiring a real key.