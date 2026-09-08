Expose these four. Each has exactly one required string param, and each has a second parameter that is a genuine free-text or short-token string a small model can plausibly fill:

  brave_web_search   query (string) [+ freshness: 'pd'|'pw'|'pm'|'py']
  brave_news_search  query (string) [+ freshness: 'pd'|'pw'|'pm'|'py']
  brave_video_search query (string) [+ freshness]
  brave_image_search query (string) [+ safesearch: 'off'|'strict']

Rationale per your Tool/Parameter contract: query maps cleanly to Parameter(name='query', freeform=True); freshness is a 4-way closed vocabulary, short enough that a 350M model copies it out of the description rather than inventing one, and it degrades safely (a bad value is rejected by zod, not silently mis-searched). Everything else on these tools is an int, a boolean, an array, or a 37-to-51-member enum -- exactly the shapes a sub-1B model fills wrong.

Borderline, expose only if you need it:
  brave_place_search  query (string) + location (string, 'san francisco ca united states')
    Two clean strings and the second is genuinely freeform, but the schema has ZERO required params. A small model will call it with an empty object and get an unfocused global POI dump. If you expose it, make query and location both required on your side even though the server tolerates their absence.

Hide these three:
  brave_summarizer   key is required but is an opaque server-generated token from a prior brave_web_search with summary=true. No model of any size authors it; it needs a two-call orchestration your harness would have to thread. Nothing for a 350M model to fill.
  brave_llm_context  query alone technically drives it, but 25 input properties is a huge schema to put in a small model's context, its purpose overlaps brave_web_search enough to cause routing confusion, and its default output (8192 tokens of extracted page text) is roughly 40x MAX_TOOL_OUTPUT_CHARS, so truncation to 800 chars leaves you with the first fragment of the first snippet.
  brave_local_search  one required string, so mechanically fine, but its input schema is byte-identical to brave_web_search's and it silently falls back to web results without a Pro plan. Two query-only tools with near-identical descriptions is the single worst thing you can do to a small model's tool-selection accuracy. Route "restaurants near me" through brave_web_search instead.

Enforce the whitelist at the server, not just in your prompt, so the model never sees the hidden tools:
  --enabled-tools brave_web_search brave_news_search brave_video_search brave_image_search
  or env BRAVE_MCP_ENABLED_TOOLS="brave_web_search brave_news_search brave_video_search brave_image_search" (space-separated; --enabled-tools and --disabled-tools cannot be combined, and an unknown name aborts startup with the valid list printed to stderr).