Shape (brave_web_search, the one you will actually wire up): the MCP result is content[] with one type:'text' block PER RESULT, each block a compact single-line JSON.stringify. There is no outputSchema and no structuredContent on this tool, so you get N separate strings, not one document:

  {"url":"https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)","title":"Transformer (deep learning architecture) - Wikipedia","description":"In deep learning, the <strong>transformer</strong> is a neural network architecture based on the multi-head attention mechanism ..."}

Notes that bite:
  - text_decorations defaults to true, so description contains literal <strong>...</strong> markers. Pass text_decorations=false, or strip the tags, before the text reaches a 350M model.
  - extra_snippets is present only when you ask for it, and adds up to 5 more excerpts (~600-1200 extra bytes per result).
  - With summary=true an extra leading block "Summarizer key: <uuid>" appears before any result.
  - Empty result set is not an exception: isError:true with one block, "No web results found".

Size, against your MAX_TOOL_OUTPUT_CHARS = 800:
  brave_web_search   default count=10 -> ~300-600 bytes per block -> 3-6 KB per call. With count=20 and extra_snippets, 15-25 KB.
  brave_news_search  default count=20 -> ~400-700 bytes per block -> 8-14 KB.
  brave_video_search default count=20 -> ~350-600 bytes per block -> 7-12 KB.
  brave_image_search default count=50, and it is ONE block holding the whole array -> ~250-350 bytes per item -> 12-18 KB in a single string. Worst offender for naive truncation: chop at 800 chars and you cut mid-JSON, leaving unparseable garbage.
  brave_llm_context  default 8192 tokens of page text -> 30-40 KB; at maximum_number_of_tokens=32768, ~130 KB.
  brave_place_search one block of the raw API response, POIs carry hours/photos/categories -> 10-50 KB.

So the default response is 4x to 30x your truncation budget; 800 chars is about 1.5 to 2.5 web results. Two consequences for the harness:
  1. Do not hand the raw MCP content array to the 800-char truncator. Re-render first, one line per result as "title -- url" plus a one-sentence clipped description, then truncate. Six results at ~120 chars each fits 800 comfortably and stays parseable at the cut.
  2. Pin count server-side rather than exposing it as a parameter -- your Tool.run can hardcode count=5 (and text_decorations=false, and for images count=8) in the JSON-RPC call while the model still only sees query. That keeps the two-string surface intact and makes truncation almost never fire.