Source read: npm tarball @brave/brave-search-mcp-server@2.1.3 (dist/ = compiled src/), cross-checked with github.com/brave/brave-search-mcp-server at tag v2.1.3 (server.json, Dockerfile, .github/workflows/build-release.yml). Extracted at /private/tmp/claude-501/-Users-awaheed-Research-AIAgents/249d0cf8-e005-408b-8151-8abc211c8b61/scratchpad/package/dist.

8 tools, registered in src/tools/index.js in this order. Every one is registered unconditionally unless filtered by --enabled-tools/--disabled-tools (src/config.js isToolPermittedByUser). All input schemas are zod; all take a flat object of scalars (no nesting except brave_summarizer's unused chat variant). Types below are the zod types, i.e. what the JSON Schema advertises.

1. brave_web_search  (src/tools/web/, params in web/params.js)
   REQUIRED: query: string, max 400 chars, refined to <=50 words.
   OPTIONAL: country: enum of 37 codes + 'ALL' (default 'US'); search_lang: enum of 50 lang codes (default 'en'); ui_lang: enum of 38 locale codes (default 'en-US'); count: int 1-20 (default 10, web results only); offset: int 0-9 (default 0); safesearch: enum 'off'|'moderate'|'strict' (default 'moderate'); freshness: 'pd'|'pw'|'pm'|'py' OR a string matching /^\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}$/; text_decorations: boolean (default true); spellcheck: boolean (default true); result_filter: array of enum ['discussions','faq','infobox','news','query','summarizer','videos','web','locations','rich'] (default ['web','query']) -- note the tool's own description misspells this as "results_filter", the real key is result_filter; goggles: string | string[]; units: 'metric'|'imperial'; extra_snippets: boolean; summary: boolean (must be true to later use brave_summarizer).
   RETURNS: content[] of type:'text' blocks, no outputSchema/structuredContent. One block per web result, JSON.stringify({url,title,description,extra_snippets}). If summary=true, a first block "Summarizer key: <key>". If result_filter widens, extra blocks are appended for FAQ ({question,answer,title,url}), discussions ({mutated_by_goggles,url,data}), news, and videos. Zero web results => isError:true and a single "No web results found" block.

2. brave_local_search  (src/tools/local/index.js)
   Input schema is web/params.js verbatim (imported as webParams), so: REQUIRED query: string; OPTIONAL identical to brave_web_search.
   RETURNS: content[] of text blocks, JSON.stringify({name,price_range,phone,rating,hours,rating_count,description,address}) per POI. Implementation does a web search, takes up to 20 location ids, then calls /res/v1/local/descriptions. Requires a Brave Pro plan; with no locations it silently falls back to formatted web results prefixed with an explanatory text block.

3. brave_video_search  (src/tools/videos/params.js)
   REQUIRED: query: string, min 1, max 400, <=50 words.
   OPTIONAL: country: string (default 'US'); search_lang: string (default 'en'); ui_lang: string (default 'en-US'); count: int 1-50 (default 20); offset: int 0-9 (default 0); spellcheck: boolean (default true); safesearch: 'off'|'moderate'|'strict' (default 'moderate'); freshness: 'pd'|'pw'|'pm'|'py' | 'YYYY-MM-DDtoYYYY-MM-DD'.
   RETURNS: content[] text blocks, JSON.stringify({url,title,description,duration,thumbnail_url}) per video. No outputSchema.

4. brave_news_search  (src/tools/news/params.js)
   REQUIRED: query: string, max 400, <=50 words (no min(1) here).
   OPTIONAL: country/search_lang/ui_lang: plain strings with defaults 'US'/'en'/'en-US'; count: int 1-50 (default 20); offset: int 0-9 (default 0); spellcheck: boolean (default true); safesearch: 'off'|'moderate'|'strict' (default 'moderate'); freshness: same union; extra_snippets: boolean (default false); goggles: string | string[].
   RETURNS: content[] text blocks, JSON.stringify({url,title,age,page_age,breaking,description,extra_snippets,thumbnail}) per article. No outputSchema.

5. brave_image_search  (src/tools/images/schemas/input.js + output.js)
   REQUIRED: query: string, min 1, max 400, <=50 words.
   OPTIONAL: country: string (default 'US'); search_lang: string (default 'en'); count: int 1-200 (default 50); safesearch: 'off'|'strict' only (default 'strict', no 'moderate'); spellcheck: boolean (default true).
   RETURNS: has an outputSchema. ONE text block containing JSON.stringify of the whole payload, plus the same object as structuredContent: {type:'object', items:[{title,url,page_fetched (ISO datetime),confidence,properties:{url,width,height}}], count, might_be_offensive}. Items failing the simplified schema are dropped; a whole-payload parse failure sets isError:true and returns zod's flattened error instead.

6. brave_summarizer  (src/tools/summarizer/params.js -> summarizerQueryParams)
   REQUIRED: key: string -- the opaque summarizer key returned by a prior brave_web_search with summary=true.
   OPTIONAL: entity_info: boolean (default false); inline_references: boolean (default false).
   (params.js also exports chatCompletionParams with messages/model/stream/country/language/enable_entities/enable_citations, but that schema is NOT registered on any tool.)
   RETURNS: a single text block of concatenated summary tokens; inline_reference parts render as " (url)". Polls /res/v1/summarizer/search up to 20 times at 50ms. On failure/empty: isError:true, "Unable to retrieve a Summarizer summary." Needs a Pro AI subscription.

7. brave_llm_context  (src/tools/llm_context/schemas/input.js -> LlmContextInputSchema = RequestParamsSchema + RequestHeadersSchema)
   REQUIRED: query: string, trimmed, min 1, max 400, <=50 words.
   OPTIONAL params: country: enum 37 codes + 'ALL'; search_lang: enum 51 codes; count: int 1-50 (default 20); spellcheck: boolean; maximum_number_of_urls: int 1-50; maximum_number_of_tokens: int 1024-32768 (default 8192); maximum_number_of_snippets: int 1-256 (default 50); context_threshold_mode: 'disabled'|'strict'|'lenient'|'balanced' (default balanced); maximum_number_of_tokens_per_url: int 512-8192 (default 4096); maximum_number_of_snippets_per_url: int 1-100 (default 50); goggles: string | string[]; freshness: same union; enable_local: boolean; enable_source_metadata: boolean.
   OPTIONAL header params, flattened into the same input object: 'x-loc-lat': number -90..90; 'x-loc-long': number -180..180; 'x-loc-city': string; 'x-loc-state': string max 3; 'x-loc-state-name': string; 'x-loc-country': string length 2; 'x-loc-postal-code': string; 'api-version': string YYYY-MM-DD; accept: 'application/json'|'*/*'; 'cache-control': 'no-cache'; 'user-agent': string. That is 25 input properties, the largest surface of any tool.
   RETURNS: outputSchema. One text block of JSON.stringify(payload) plus structuredContent: {grounding:{generic:[{url,title,snippets:string[]}], poi:{name,url,title,snippets}|null, map:[{name,url,title,snippets}]}, sources: {<url>: {title,hostname,age[],site_name?,favicon?,thumbnail?}}}. snippets carry the actual extracted page text/tables/code, and entries may themselves be JSON-encoded strings.

8. brave_place_search  (src/tools/place_search/schemas/input.js -> PlaceSearchInputSchema)
   REQUIRED: none. query is optional and empty-string-transforms to undefined.
   OPTIONAL: query: string trimmed max 400, <=50 words; radius: number >=0 (meters, a bias not a cutoff); count: int 1-50 (default 20); latitude: number -90..90; longitude: number -180..180; location: string min 1 (e.g. 'san francisco ca united states'); country: enum 37 codes; search_lang: enum 51 codes; ui_lang: enum 38 locales; units: 'metric'|'imperial'; safesearch: 'off'|'moderate'|'strict'; spellcheck: boolean; geoloc: string; plus the same four header params 'api-version', accept, 'cache-control', 'user-agent'.
   RETURNS: outputSchema. One text block of JSON.stringify(response) plus the raw API response as structuredContent, any combination of results (POIs with postal address, hours, contact, ratings, photos, categories, timezone), cities, addresses, streets, and location.

Server-level facts: server name 'brave-search-mcp-server', instructions "Use this server to search the Web for various types of data via the Brave Search API."; capabilities logging + tools{listChanged:false}; schemas are forced to JSON Schema 2020-12 (src/jsonSchemaDialect.js) rather than the SDK's default draft-07. Endpoints hit: /res/v1/{web,news,videos,images,local/pois,local/descriptions,summarizer,llm/context,local/place_search}/search under api.search.brave.com, auth via X-Subscription-Token. src/constants.js declares RATE_LIMIT {perSecond:1, perMonth:15000}, though the enforcement call is commented out.