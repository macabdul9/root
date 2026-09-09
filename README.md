# root

Small tool-using agents that run entirely on a local sub-billion-parameter open-weight
model. Lowercase, because it is smol.

The default model is `LiquidAI/LFM2.5-350M`. Four others are configured, and any instruct
model whose chat template accepts a `tools` argument works via `--model` or `/model`.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/macabdul9/root/main/install.sh | bash
```

The installer uses [uv](https://docs.astral.sh/uv/), fetching it first if you do not have it.
uv builds an isolated environment for root and puts `root` and `root-eval` launchers in
`~/.local/bin`; your system and conda Pythons are untouched, and uv picks a suitable Python
itself if none is installed. Most of the download is PyTorch. Model weights are fetched on
first run, into `~/.cache/huggingface`.

Then, from any directory:

```bash
root                                  # interactive session
root "What is 17 times 23, plus 4?"   # one-shot
root --init                           # copy the configs into ./configs to edit
```

| Variable | Effect |
| --- | --- |
| `ROOT_BIN` | Where the launchers go (default `~/.local/bin`) |
| `ROOT_HOME` | Where the environment lives (default is uv's tool directory) |
| `ROOT_REF` | Branch or tag to install (default `main`) |

Re-running the installer upgrades in place. `install.sh --uninstall` removes both launchers
and the environment, leaving the weights cache alone. If you already use uv, the installer is
only a wrapper around one command:

```bash
uv tool install "root @ git+https://github.com/macabdul9/root"
```

## From a checkout

```bash
git clone https://github.com/macabdul9/root && cd root
scripts/setup.sh
```

`scripts/setup.sh` runs `uv sync --extra dev`, which creates `.venv` from `uv.lock` so every
checkout resolves to the same versions. Constrained tool calls need `outlines`, which `dev`
includes; `uv sync --extra grammar` is the same dependency without the test tooling, and
without it root parses tool calls instead of constraining them. Run things through uv rather than activating:

```bash
uv run root                  # the terminal
uv run pytest                # the tests
uv run ruff check src tests  # the linter
scripts/evaluate.sh          # the eval suite, into runs/
```

`source .venv/bin/activate` still works if you prefer it. Adding a dependency is
`uv add <package>`, which updates `pyproject.toml` and the lockfile together.

`install.sh` run from inside a checkout installs that working tree instead of fetching from
GitHub, which is the quick way to try a local change as an installed command.

To build once and run or test on other machines with Docker or Apptainer, see
[the container guide](docs/containers.md). The image includes the locked runtime,
grammar support, and test tools, and defaults to NVIDIA GPU inference with CUDA.
On an HPC machine with only Apptainer, build directly from the checkout with
`apptainer build --fakeroot root.sif containers/root.def`; Docker is not required.
Run `./scripts/container.sh` to build if needed, test, and start the NVIDIA GPU
terminal using this checkout as the workspace and your existing Hugging Face
cache. It detects Apptainer or Docker, preferring Apptainer when both are installed.
Set `ROOT_CONTAINER_RUNTIME=docker` or `ROOT_CONTAINER_RUNTIME=apptainer` to choose
explicitly. Docker needs NVIDIA Container Toolkit configured; on HPC, run inside
your GPU allocation.

## Configuration

`root` reads `agents.yaml`, `models.yaml` and `evals.yaml` in that order of preference:
an explicit `--config`, then a `configs/` directory beside your working directory, then the
copies shipped inside the package. That is what lets the installed command work from
anywhere, while a project with its own `configs/` still overrides the defaults. `root --init`
writes the packaged copies into `./configs` to start from, and leaves any file already there
alone.

## The terminal

`root` with no prompt opens an interactive session. The model loads once and stays loaded,
which is the difference between 4 seconds a question and 4 seconds plus 15 of startup.

```
$ root
Loading LiquidAI/LFM2.5-350M...
root · LFM2.5-350M · auto · /help for commands, ctrl-d to leave
root> /agent calc
agent: calc
root> What is 17 times 23, plus 4?
  ● calculator('17 * 23 + 4')
    ⎿ 395
      0.4s · 15 tok · tool 0.0s · forced
The result of 17 times 23 plus 4 is 395.
[0.7s · 31 tok · 1 call]
root> Now subtract 95 from that.
  ● calculator('17 * 23 - 95')
    ⎿ 296
      0.3s · 15 tok · tool 0.0s · forced
The result of subtracting 95 from 296 is 301.
[0.6s · 30 tok · 1 call]
```

The prompt is always `root>`; the banner and `/agents` say which agent is active. The second
turn shows both the memory working and its limit: `that` resolves to the previous question,
but the model then narrates the tool result wrongly, calling 296 the answer to `301`.

| Command | Effect |
| --- | --- |
| `/agent`, `/agents` | With a name, switch agent, clearing the conversation (`auto` to route); without one, list them |
| `/tools` | Show what the current agent may call |
| `/routes` | Show the rules `auto` uses to pick an agent |
| `/agent`, `/model`, `/trace` | With no argument, list the options and how to set them |
| `/model [NAME]` | List models, or load one by alias or Hugging Face id |
| `/inference [NAME]`, `/engine` | Show or switch the engine: `auto`, `transformers`, `vllm`, `sglang`, `ollama` |
| `/trace [LEVEL]` | Show or set the trace level: `off`, `on`, `full` |
| `/stream [on\|off]` | Stream the answer as it is generated |
| `/decoding [K=V]` | Show or set temperature, top_p, top_k, min_p, max_new_tokens |
| `/last` | Replay the previous turn in full detail |
| `/workspace [PATH]` | Show or change where the file tools look |
| `/new` | Forget the conversation so far |
| `/help`, `/quit` | The obvious |

The last four exchanges carry into the next prompt; the tool calls behind them do not, since
replaying old observations into a 350M context invites it to answer from a stale one. Line
editing and history come from readline, kept in `~/.root_history`.

## Models

`configs/models.yaml` lists what `/model` offers and names the default in its `default:` key,
which is the only place the default lives. `/model` alone shows the list, `/model qwen3-06b`
loads one, and any Hugging Face id works whether listed or not. The old weights are freed
before the new ones load, and the conversation is cleared, since it was written by a
different model.

```
root> /model
* lfm2-350m                350M, the default
  lfm2-230m                230M, the smallest
  qwen3-06b                600M, JSON tool calls
  qwen35-08b               800M, XML tool calls
  k2-horizon-09b           900M, runs code from its own repo
  qwen3-coder-30b-a3b      30.5B MoE, tuned for code, XML tool calls, served engine only
  glm-4.7-flash            31.2B MoE, GLM tool calls, served engine only
root> /model qwen3-06b
model: Qwen/Qwen3-0.6B · tool calls: qwen-json
```

Any Hugging Face model can be added to the list and kept there:

```
root> /model add HuggingFaceTB/SmolLM2-135M-Instruct
registered smollm2-135m-instruct -> HuggingFaceTB/SmolLM2-135M-Instruct
model: HuggingFaceTB/SmolLM2-135M-Instruct · tool calls: generic
root> /model remove smollm2-135m-instruct
```

`/model add ORG/NAME [alias]` checks the id against the hub before writing anything, so a
typo costs a second rather than a failed download, and it records the parameter count as the
note. The entry is written to `./configs/models.yaml`, never into the installed package:
registration creates that file from the packaged defaults on first use, which is the same
directory `config_path` already prefers. Comments in the file survive the edit. Passing an
alias is optional; without one, `LiquidAI/LFM2.5-350M` becomes `lfm2.5-350m`.

A model whose chat template has no tool-call markers loads under the `generic` format, which
means the toolless agents work and a forced tool call is not available. `/model` shows the
format each model resolved to when you load it.

## Eval findings

`configs/evals.yaml` holds 114 cases across every agent. Each model runs all of them through
the same agents and tools. Tool choice and answer content are scored separately, because they
fail separately. Cases that touch files use a fixture workspace shipped with the package, and
cases that write files get a fresh temporary directory, so the suite scores the same from a
checkout or from an installed command in an unrelated directory.

```bash
uv run root-eval                     # all 114; about 4 minutes on the default model
uv run root-eval --agent python      # one agent
uv run root-eval --model qwen35-08b  # one model
```

| Model | Cases | Tool choice | Answer content |
| --- | --- | --- | --- |
| `LiquidAI/LFM2.5-230M` | 91/114 | 96% | 82% |
| `LiquidAI/LFM2.5-350M` (default) | 97/114 | 97% | 88% |
| `Qwen/Qwen3-0.6B` | 96/114 | 96% | 87% |
| `Qwen/Qwen3.5-0.8B` | **103/114** | 98% | 92% |
| `IFM/K2-Horizon-0.9B` | 99/114 | 94% | 92% |

K2-Horizon is the one that moves between runs, scoring 95 and 97 on two passes of the same
suite. Both passes were identical except in `code` and `chat`, which are the only two agents
that sample; every temperature-0 agent gave the same answers both times, and the default
model repeats its `now` score exactly whether run alone or inside the full suite. So the
spread is sampling, not flakiness in the harness.

Per agent, on the default model:

| Agent | LFM2.5-350M | Qwen3.5-0.8B | K2-Horizon-0.9B |
| --- | --- | --- | --- |
| `calc` | 15/15 | 15/15 | 15/15 |
| `code` | 7/7 | 7/7 | 7/7 |
| `pdf` | 3/3 | 3/3 | 3/3 |
| `chat` | 9/10 | 10/10 | 10/10 |
| `write` | 10/11 | 10/11 | 10/11 |
| `extract` | 8/8 | 8/8 | 6/8 |
| `data` | 4/4 | 4/4 | 3/4 |
| `search` | 9/10 | 9/10 | 9/10 |
| `files` | 16/17 | 16/17 | 13/17 |
| `now` | 7/7 | 6/7 | 4/7 |
| `repo` | 2/2 | 2/2 | 1/2 |
| `python` | 13/20 | 13/20 | 18/20 |

What the numbers showed, each of which changed the code:

- **Choosing the tool is easy; reading its output is not.** Tool choice sits at 94-98% for
  every model including the 230M one, while answer content spreads from 82% to 92%. Nothing
  in this project is bottlenecked on picking the right tool any more.
- **The models differ far more by agent than the totals suggest.** K2-Horizon scores 18/20 on
  `python`, where every other model sits at 13/20 and cannot write a correct program; the
  same model is the worst of the five at `files` (13/17) and `now` (4/7). A single number per
  model hides that completely, which is why the per-agent table is here.
- **Constrained decoding was worth less than expected.** Restricting a forced call to a
  grammar was the highest-value item on the roadmap, on the theory that it would take tool
  choice to 100% and let three parse repairs be deleted. It did neither. Tool choice was
  already 94-98% and the remaining failures are answer content, which a grammar cannot
  touch; every repair is still needed for unconstrained steps and for the served engines.
  What it did cost was roughly four times the eval runtime and, until the index cache was
  bounded, enough memory to have runs killed. It is on by default because it does help the
  structured cases, and `--no-grammar` is there because that margin is thin.
- **A weakness can be specific to one model and one tool.** K2-Horizon scores 4/7 on the
  clock agent, and always the same three: asked for a relative date it calls `today` with an
  empty offset, emitting `<ifm|arg_value></ifm|arg_value>` rather than `+3 days`. The parse
  is correct and the other four models handle it, so this is that model, not the harness.
- **`run_python` is where the ceiling is.** The `python` agent scores 13/20 on the default
  model while five of the twelve agents are perfect. Writing a correct program is simply harder
  than filling in one string argument, and the failures are all arithmetic or string
  reasoning the model got wrong inside the code it wrote.
- **Schema text is load-bearing across tools.** `read_file`'s parameter description used a
  single nested example, `src/app.py`. The model copied the shape and asked for
  `src/CHANGELOG.md` and `src/README.md` for files at the workspace root. Changing the
  example to `README.md or src/app.py` took `files` from 12/15 to 15/15, without touching
  any prompt. Earlier, rewording `list_files` had changed what `read_file` was called with in
  a different case. Tool descriptions are one prompt; change one and re-run the suite.
- **A general agent is worse than a routed specialist.** One agent holding every tool scored
  3/8 with tool choice at 62%, against 100% for specialists. Asking the model to route
  instead scored 0/12, so `route.py` uses regular expressions.
- **Small models slip in predictable ways worth repairing.** A missing `print()`, a trailing
  `)`, a tool call with unbalanced quotes. Each was a total failure until it was handled, and
  each is deterministic enough to fix in `tools.py` and `formats.py` rather than by prompting.

Writing the cases also caught four of my own that were unfair rather than hard: a prompt that
grepped for `rank`, which is a substring of `ranking` in four unrelated files; one asking
which file "imports the pipeline module" when a shell script referenced it too; an extract
case beginning "Bill from Acme Corp", where `Bill` reads as a name; and a definition question
scored on a word the correct answer need not contain. An eval case that a careful reader
could answer two ways measures nothing.

Scores move by a point or two between runs, and the movement is confined to `chat`
(temperature 0.3) and `code` (0.2). Everything else decodes greedily and repeats exactly,
which was checked by running one agent alone and comparing it against the same agent inside
the full suite.

`IFM/K2-Horizon-0.9B` ships a custom architecture, so loading it executes Python from its
repository. Its entry carries `trust_remote_code: true` and nothing else does; deleting that
line disables the model rather than silently running the code.

## Tool-call formats

The listed models write tool calls five different ways, so `formats.py` holds one `CallFormat`
per family and `detect_format` picks between them by looking for a marker in the model's own
chat template. Reading the template beats a per-model table: a model this project has never
heard of still works if it writes calls like one it has.

| Format | Shape |
| --- | --- |
| `lfm2` | `<\|tool_call_start\|>[calculator(expression='6 * 7')]<\|tool_call_end\|>` |
| `qwen-json` | `<tool_call>{"name": "calculator", "arguments": {...}}</tool_call>` |
| `qwen-xml` | `<tool_call><function=calculator><parameter=expression>6 * 7</parameter>...` |
| `ifm` | `<ifm\|tool_calls><ifm\|tool_call>{"name": ...}</ifm\|tool_call>...` |
| `glm` | `<tool_call>calculator<arg_key>expression</arg_key><arg_value>6 * 7</arg_value>...` |
| `generic` | A bare `calculator('6 * 7')` line, with no way to force a call |

Order matters in `KNOWN`: `qwen-json` is tried last, because every other `<tool_call>`
template contains its marker too, and only the extra `<function=` or `<arg_key>` tells them
apart. GLM is also the one format whose marker is not in its prefill - a forced turn opens
`<tool_call>`, and `<arg_key>` only appears once the model has written the tool's name.

Each format declares the markers its parser needs in `keep`. That matters because decoding
strips control tokens: K2-Horizon writes arguments inside `<ifm|arg_value>` tags, which live
in the tokenizer's added vocabulary, and stripping them leaves a call the parser reads as no
call at all.

Reasoning models need one more step. Qwen3 emits `<think>...</think>` even with thinking
disabled in the template, and K2-Horizon's generation prompt opens its `<ifm|think>` block so
the model emits only the closing tag. `strip_thinking` handles both, plus the truncated case,
and only for the answer: the trace keeps the raw generation.

## Routing

The session starts on `auto`, which picks an agent from the wording of each prompt and shows
what it picked:

```
root> [2,3,41,0,19,-10] sort this number
  ● route → python (list literal)
  ● run_python('sorted([2,3,41,0,19,-10]))')
    ⎿ [-10, 0, 2, 3, 19, 41]
```

`/agent chat` pins one agent and turns routing off; `/agent auto` turns it back on.
`--agent auto` is also the default for one-shot runs.

The rules are plain regular expressions in `route.py`, in order, first match wins. That is a
deliberate choice over the two alternatives, both of which were measured and lost:

- **One agent holding every tool** scores 3/8 (tool choice 62%) where the specialists score
  10/11 (tool choice 100%). Given eight tools it answers file and code questions from memory
  without calling anything, and when it does call, it wanders: `list_files → file_info →
  file_info → run_python`, then invents a filename.
- **Asking the model to route** scores 0/12. It replies with a single clean handler name,
  just the wrong one: `search` for "What is 17 times 23, plus 4?", `extract` for "Explain why
  a 350M model is faster than a 7B one". Classification is not free at this size.

On sixteen prompts written after the rules were finished, the router places 9. That is the
number to trust; the rules score 16/16 on the prompts they were written against, which means
nothing. What makes 56% acceptable is the shape of the other 44%: **every** one of those was
a rule failing to fire, not a rule firing wrongly. Across 32 held-out prompts there were no
misroutes at all, and a test asserts that property. A miss lands you on `chat`, which is
where you would have been anyway, and the trace says so:

```
root> who are you?
  ● route → chat (no rule matched)
I'm a language model developed by Liquid AI.
[0.7s · 51 tok · 0 calls · agent has no tools]
  no rule matched, and chat has no tools - try /agent python
```

That hint is there because a toolless agent asked to do work is the single most confusing
thing this project does. Rules are cheap to add: a line in `RULES` and a case in
`tests/test_route.py`.

## Inference engines

Four engines generate tokens; the agent loop does not care which. `/inference` switches the
running session, `--engine` sets it for one run, and the default `auto` picks the fastest
engine that is already serving the model, falling back to `transformers`.

```
root> /inference
inference: transformers · switch with /inference auto vllm sglang ollama
root> /inference ollama
inference: ollama · ollama @ http://127.0.0.1:11500
```

| Engine | How it runs | Needs |
| --- | --- | --- |
| `transformers` | In-process PyTorch, CUDA/MPS/CPU | Nothing; this is what `install.sh` sets up |
| `ollama` | `ollama serve`, llama.cpp with Metal or CUDA | Ollama running; the model is pulled automatically |
| `vllm` | `vllm serve MODEL`, OpenAI completions API | A vLLM server; on Apple silicon add [vllm-metal](https://github.com/vllm-project/vllm-metal) or it runs on CPU |
| `sglang` | `sglang.launch_server`, same API | An SGLang server, Linux only |
| `tokenspeed` | `tokenspeed serve MODEL`, same API | A [TokenSpeed](https://github.com/lightseekorg/tokenspeed) server, Linux and CUDA only |

Ollama names models its own way, so root matches an installed tag against the Hugging Face
id and, failing that, pulls one: first the library tag it can derive
(`Qwen/Qwen2.5-Coder-0.5B-Instruct` becomes `qwen2.5-coder:0.5b`), then `hf.co/<id>`, which
works for a repository holding GGUF files. Probing never pulls: `auto` asks every engine
whether it is already serving the model, and a question should not start a download.

Every engine is given a prompt this project rendered itself, not a message list. The model's
own chat template, the tool schemas, the forced-call prefill and the stop markers are applied
locally and the server is asked only to continue text, so an agent behaves identically
whichever engine is underneath. Point at a server elsewhere with `ROOT_OLLAMA_URL`,
`ROOT_VLLM_URL`, `ROOT_SGLANG_URL`, or `--engine-url`.

### Throughput

`uv run root-bench` measures generation, 256 tokens per run, after a warm-up. Measured on an
M-series MacBook, macOS 26.4, `Qwen/Qwen3-0.6B`.

One request at a time, which is what a terminal session does:

| Engine | tok/s | First token | Backend used here |
| --- | --- | --- | --- |
| `ollama` | **94-113** | 0.02s | llama.cpp, Metal, Q4_K_M |
| `transformers` | 34.4 | 0.07s | PyTorch, MPS, bfloat16 |
| `vllm` + vllm-metal | 29.9 | 0.74s | MLX, Metal, bfloat16 |
| `vllm` core only | 24.5 | 0.15s | PyTorch, **CPU**, bfloat16 |
| `sglang` | not measurable | - | Linux-only wheels |

Eight requests in flight, which is what a serving engine is built for:

| Engine | tok/s | First token |
| --- | --- | --- |
| `vllm` + vllm-metal | **192.6** | 0.22s |
| `ollama` | 110.0 | 4.01s |
| `transformers` | cannot | - |

The ordering inverts, and the second table is the one that reflects each engine's design.
vLLM batches concurrent requests through one paged KV cache, so eight streams cost little
more than one and time-to-first-token stays at 0.22s. Ollama queues them: aggregate
throughput barely moves and the eighth caller waits four seconds to see anything. The
in-process engine cannot do it at all — several threads calling `generate` on one model abort
the process on an MPS command-buffer assertion, so `root-bench` refuses that combination
rather than crashing.

Three more caveats:

- **Ollama is not running the same numbers.** Its `qwen3:0.6b` tag is 4-bit quantized, about
  a third the size of the bfloat16 weights the other engines load. Some of its single-stream
  lead is the engine and some is the quantization; this measurement does not separate them.
- **Core vLLM has no Metal backend.** `vllm/platforms/` ships `cpu`, `cuda`, `rocm`, `tpu`
  and `xpu`; on Apple silicon it detects `cpu` while torch reports MPS available, and runs on
  CPU cores with the GPU idle. The
  [vllm-metal](https://github.com/vllm-project/vllm-metal) plugin adds one through vLLM's
  hardware-plugin interface, lowering the model onto MLX. It is a separate install and worth
  it: `Set Metal wired_limit to 17.8 GB` in the server log is how you know it took.
- **TokenSpeed is unmeasured here.** Its server speaks the same OpenAI completions API, so
  root needed no code beyond registering it, but `tokenspeed-kernel` pulls CUDA-only
  manylinux wheels and will not install on macOS. It also defaults to port 8000, the same as
  vLLM, so `auto` cannot tell the two apart by probing: they take the same code path, so the
  only cost is the label. Set `ROOT_TOKENSPEED_URL` if you run both.
- **SGLang cannot run on macOS at all.** `sglang` 0.5.19 and `sgl-kernel` publish only
  `manylinux` wheels, and asking uv for it on darwin resolves back to a 2024-era version with
  no server in it. The engine is implemented and will work against a Linux server, but no
  honest number can go in that row from this machine.

`auto` prefers ollama, then vllm, then sglang, then tokenspeed, then transformers, and skips any that is not
already serving the model. That order follows the single-stream table, because root issues
one request at a time; a deployment serving many users at once should pin `--engine vllm`
instead. On a machine with nothing running, `auto` is `transformers`, which is also the
engine every number in [Eval findings](#eval-findings) was measured on.

The default model is faster than the benchmark model on the same engine:
`LiquidAI/LFM2.5-350M` runs at 59.8 tok/s under `transformers`, against 34.4 for
`Qwen/Qwen3-0.6B`.

## Identity and the shared prompt

Asked who it is, root answers from a template rather than from the model:

```
root> who are you?
I am an AI agent named "root" (no, I am not scary) built by Abdul Waheed. I live in
your terminal. The underlying large language model (or rather smol) is
LiquidAI/LFM2.5-350M.

I can do arithmetic exactly, write and run a short Python program, read files in your
workspace, create and edit files, pull the text out of a PDF, search your code, answer
questions about a CSV in SQL, read your git history, and tell you the date...
```

That question never reaches the model, and the reason is measured rather than fussy: asked
who it is, LFM2.5-350M has variously claimed to be built by Naver and by "Nathaniel J.
Hughes and Microsoft". The model name and the ability list are assembled from what is
actually loaded, so switching model or adding a tool changes the answer and it cannot go
stale. `is_identity_question` catches the variants, guarded so that "what are you going to
do with [1, 2, 3]?" stays a task.

`configs/agents.yaml` also has a `preamble:` prepended to every agent's own prompt, the
shared system prompt most agent harnesses have, with `{{model}}`, `{{agent}}`, `{{tools}}`
and `{{date}}` filled in at run time.

**It ships empty**, because at this size it costs more than it gives:

| Preamble | `python` | `search` | `files` |
| --- | --- | --- | --- |
| none | 13/20 | 8/10 | 16/17 |
| three lines | 10/20 | 6/10 | 15/17 |
| fifteen lines | 9/20 | 5/10 | 15/17 |

Six cases across the three agents it touches most, and the mechanism is specific rather than
a vague loss of attention. Diffing the flipped cases, both `search` regressions switched from
`grep` to `read_file` and both asked for the same nonexistent path, `src/app.py` — which is
the example in `read_file`'s own parameter description. Pushed further from the generation
point by three lines of preamble, the agent's instruction to use `grep` loses to the most
concrete text still nearby, and the model copies the schema's example argument instead of
answering the question.

That is the same failure this project has hit twice before: `read_file`'s example once being
only `src/app.py` made the model ask for `src/CHANGELOG.md`, and rewording `list_files`
changed what `read_file` was called with. Tool descriptions are load-bearing prompt text at
this size, and anything that displaces the task-specific instruction lets them win.

The full text is in the config, commented out, with the substitutions intact: uncomment it and
rerun `uv run root-eval` to see which way it goes on a larger model, where the same crowding
is much less likely.

## Decoding

Every agent carries its own decoding settings, in `configs/agents.yaml` under `defaults:` or
per agent:

```yaml
defaults:
  temperature: 0.3     # 0 is greedy; above 0 samples
  top_p: 0.9           # nucleus mass
  top_k: 0             # 0 is off
  min_p: 0.0           # 0 is off
  repetition_penalty: 1.0
  max_new_tokens: 256
```

The tool-using agents ship at `temperature: 0` because a tool call is not a place for
creativity; `chat` and `code` sample. Out-of-range values are rejected when the config loads,
naming the agent, rather than surfacing as a transformers warning mid-run.

Override for one run from the command line, or for the session from the terminal:

```bash
root --temperature 0 --top-k 40 --seed 11 "Name one advantage of running locally."
```

```
root> /decoding
code: temperature=0.2  top_p=0.9  top_k=0  min_p=0.0  repetition_penalty=1.0  max_new_tokens=400
root> /decoding temperature=0.9 top_k=40
```

`--seed` makes sampling reproducible: the same seed and prompt give the same answer, a
different seed gives a different one. Only the knobs that are on are passed to transformers,
so greedy decoding does not warn about unused sampling parameters.

### Constrained tool calls

A turn that is already committed to calling a tool can have its output restricted to a
well-formed call. Each `CallFormat` carries a regex describing its own call syntax, and
[outlines](https://github.com/dottxt-ai/outlines) compiles it into a logits processor, so an
invalid call becomes unreachable rather than repaired afterwards.

It applies only when `force_first_call` has already opened the turn with the call marker.
That is the one moment when a plain answer is not a legal output, so constraining costs
nothing; on any other step the model must be free to answer instead.

Two things are deliberately left unconstrained:

- **Free-form values.** A `Parameter` can be marked `freeform`, and `run_python`'s `code` is.
  The grammar covers the whole generation, so one unconstrainable branch turns it off for
  that turn. This is measured, not cautious: constrained, the `python` agent scored 11/20
  against 13 unconstrained, because quoting rules crowd out the code.
- **Served engines.** vLLM, SGLang, TokenSpeed and Ollama generate in their own process, so
  the grammar is a transformers-only feature. Their calls are parsed and repaired as before.

`--no-grammar` turns it off, on both `root` and `root-eval`. Worth knowing what it costs
before leaving it on: a full eval run went from about three minutes to eleven on
`Qwen3.5-0.8B`, because compiling an index over a large vocabulary is not cheap and there is
one per agent. Only the two most recent are kept; caching all twelve exhausted memory and got
several runs killed outright.

The parse repairs are all still there. Constrained decoding narrows when they are needed; it
does not replace them.

## Streaming

The terminal streams answers by default; `--stream` does the same for a one-shot run.

````
root> Write a Python function that reverses a list.
```python
def reverse_list(lst):
    return lst[::-1]
```
````

Generation runs on a worker thread while the main one drains the stream. Two things have to
be filtered on the way out, both learned by watching it leak:

- **Tool calls must not reach the screen.** A step is only known to be an answer once it is
  clear it is not a call, so `StreamGate` holds back any tail that could still become the
  format's opening marker, and drops everything once the marker arrives. A turn that was
  forced into a tool call is not streamed at all, since it is a call by construction.
- **Control tokens must not reach the screen.** The stream keeps special tokens so the gate
  can see a marker coming, which means `<|im_end|>` and its equivalents arrive too. They are
  stripped in the same place, using the same protected set the non-streaming path uses.
- **Reasoning must not reach the screen.** A thinking model streams its scratchpad first, and
  K2-Horizon's template opens the block in the generation prompt, so the stream begins inside
  one with only a closing tag to mark the answer. Whether a model does that is detected once
  at load by rendering a probe prompt, and the gate starts suppressed for those, resuming at
  the closing tag.

`/stream off` returns to printing the answer at the end, which is the only mode that applies
the code-block formatting described below; a stream is raw by nature.

## The trace

Every step prints as it finishes, so a slow call is visible rather than a hang, and a wrong
answer can be traced to the step that produced it. `--trace` sets the level for one-shot
runs, `/trace` inside a session.

At `on`, the default, each tool call shows its argument, its result, and what the step cost:

```
root> what is 2+2
  ● run_python('2+2')
    ⎿ 4
      0.3s · 12 tok · tool 0.0s · forced
The sum of 2 and 2 is 4.
[0.6s · 25 tok · 1 call]
```

`forced` means the turn was opened with the tool-call marker rather than chosen freely. The
closing line totals the turn, and says `agent has no tools` when there were none to call:

```
root> [2,3,41,0,19,-10] sort this number
The number is 19.
[0.2s · 7 tok · 0 calls · agent has no tools]
```

That line is the answer to the most common confusion here. A toolless agent asked to do
work has nothing but its own weights, and at 350M those weights cannot sort a list. Switch
to `/agent python` and the same prompt runs through `run_python`.

At `full`, every step also shows the raw generation behind it, including the tool-call
markers and anything the model wrote after them:

```
  ● run_python('2+2')
    ⎿ 4
      0.3s · 12 tok · tool 0.0s · forced
      raw: <|tool_call_start|>[run_python(code="2+2")]<|tool_call_end|>
  ● answer · 0.3s · 13 tok
      raw: The sum of 2 and 2 is 4.
```

Startup is quiet: a spinner while the weights load, then the banner. `-v` brings back the
model loading lines, the weight-loading bar and the per-step logs, which is what you want
when something is wrong rather than every time. The spinner goes to stderr and only animates
for a terminal, so `root "…" > answer.txt` still writes nothing but the answer.

`/last` reprints the previous turn at `full` regardless of the current level, which is the
usual way to ask "what did it actually generate there?" after a surprising answer.

Answers are laid out for a terminal: fenced blocks are indented and coloured, `**bold**` and
`` `code` `` become escape codes rather than punctuation. These models mix prose and code
freely, so the two need to look different.

`result.json` in a run directory carries the same steps, with per-step timings, token counts
and tool results.

## One-shot

```bash
scripts/run_agent.sh calc "A box holds 24 pens. I buy 7 boxes and give away 13. How many are left?"
scripts/run_agent.sh python "What is the sum of the squares of the numbers 1 through 20?"
scripts/run_agent.sh search "Which file defines the run_agent function?"
MODEL=LiquidAI/LFM2-1.2B scripts/run_agent.sh chat "Explain KV caching in two sentences."
```

Each invocation writes `runs/<timestamp>-<agent>/` containing the config used, `pip freeze`,
the console log, and `result.json` with the full message transcript and every tool call.

`root` is the same thing without a run directory, and reads a piped prompt from stdin:

```bash
root --list
root --agent calc --model LiquidAI/LFM2.5-350M "17 * 23 + 4"
echo "Invoice from Dana Ruiz, 2026-03-14, 412.50 USD." | root --agent extract
```

## Agents

Agents are entries in `configs/agents.yaml`: a system prompt, a list of tools, and
generation settings. Adding one is a config change, not a code change.

| Agent | Tools | Purpose |
| --- | --- | --- |
| `auto` | routes | Picks one of the below from the prompt |
| `chat` | none | Short direct answers, and honest about having no tools |
| `code` | none | Writes code for you to read, without running it |
| `now` | today | The current date and time |
| `calc` | calculator | Arithmetic word problems |
| `python` | run_python | Anything a short program answers better than the model |
| `data` | csv_query, list_files | Questions about a CSV, answered in SQL |
| `files` | read_file, read_lines, list_files | Questions about files in the workspace |
| `pdf` | read_pdf, list_files | Questions about PDF documents |
| `repo` | git, list_files | Questions about the git repository |
| `shell` | run_bash, list_files | One shell command, for what no tool covers |
| `write` | write_file, append_file, move_file, delete_file, make_directory, list_files | Creating and organising files |
| `search` | grep, read_file | Finding where something is defined or used |
| `extract` | none | Text to a fixed JSON object |

## Tools

| Tool | Argument | Returns |
| --- | --- | --- |
| `calculator` | expression | The number, via an AST walk (no `eval`) |
| `run_python` | code | Whatever the program printed |
| `csv_query` | path, query | One SQL `SELECT` over a CSV, table named `data` |
| `read_file` | path | File contents, truncated at 800 characters |
| `read_lines` | path, lines | A numbered span, `40-80`, for past the truncation |
| `read_pdf` | path | The text layer of a PDF, first 20 pages |
| `write_file` | path, content | Creates or overwrites a workspace file |
| `append_file` | path, content | Adds to the end of one, creating it if missing |
| `move_file` | source, destination | Renames or moves, refusing to overwrite |
| `delete_file` | path | Removes one file, never a directory |
| `make_directory` | path | Creates a directory and its parents |
| `run_bash` | command | Whatever a shell command printed |
| `git` | command | One read-only git subcommand, allowlisted |
| `list_files` | glob or directory | Matching workspace paths |
| `grep` | pattern or regex | `path:line: text` for each match, capped at 30 |
| `file_info` | path | Line count, byte size, modification time |
| `json_get` | `file.json:dotted.path` | One value out of a JSON file |
| `today` | empty or `+3 days` | Current date and time |

File tools resolve paths inside the workspace and refuse anything that escapes it.
`write_file` overwrites without asking and says which it did, so point a session at a
directory you are willing to have rewritten. `delete_file` takes files only: a directory can
hold work the model never saw, and removing a tree on a small model's say-so is not a risk
worth taking. `git` is an allowlist — `status`, `log`, `diff`, `branch`, `show`, `remote`,
`blame` — not a shell, so the model can read the repository but not rewrite it. `read_pdf`
extracts a text layer and does no OCR; a scanned document comes back saying so rather than
returning nothing.

Nearly every tool takes one string argument, which is what these models get right most
reliably. `write_file` takes two, and each call format expresses that natively, so a model
can write `write_file(path='a.txt', content='hi')` or name the arguments in any order.

`csv_query` loads a CSV into in-memory SQLite and runs one `SELECT` or `WITH` against it;
anything that writes is refused. It exists because these models miscount: asked how many
words are in a sentence, the default model answered 43 where the answer was 9. SQL does the
counting exactly, the way `calculator` already does the arithmetic.

`run_python` executes model-written code in a child interpreter with a 10 second timeout.
That is isolation, not a sandbox: the program runs as you, in the workspace directory, with
network access. Give the `python` agent a workspace you would hand to a stranger, or drop
the tool from the agent's list.

`run_bash` has the same standing and the same warning, plus a refusal list. It will not run
anything matching `sudo`, an `rm` reaching outside the workspace, `mkfs`/`dd`/`shutdown`, a
download piped into a shell, `git push` or `reset --hard`, a fork bomb, or `crontab`. That
list is not a security boundary — a shell has a thousand ways around a pattern — but the
accidents a small model actually has are ordinary, and these are them. Anything genuinely
untrusted wants a container, not a regular expression. If you would rather not have a shell
at all, remove `run_bash` from the `shell` agent in `agents.yaml` and it is gone.

## How the loop works

`run_agent` generates one assistant turn, looks for a tool call, runs it, and appends the
result as a `tool` message, up to `max_steps`. A turn with no tool call is the answer. Tool
errors go back to the model as the observation, so a bad argument costs a step rather than
crashing the run.

Four details exist specifically because the model is small:

- **`force_first_call: true`** opens the assistant turn with the tool-call marker so the
  model has to complete a call. Without it a 350M model answers file questions from memory
  and invents the contents.
- **One string argument per tool.** The JSON schema and the parse of `read_file(path='x')`
  both stay small enough for the model to get right.
- **Lenient call recovery.** `run_python(code="print(len("x"))")` is not valid Python, and
  the model writes it constantly. When the strict parse fails, the call is recovered by
  taking everything between the outer parentheses.
- **`ensure_print`.** A bare expression like `sum(range(1, 21))` is wrapped in `print()`.
  Otherwise the tool returns nothing and the model invents a plausible number instead.
- **`balance_parentheses`.** The model writes `sorted([2, 3, 41]))` often enough to matter,
  and cannot recover from it: shown the SyntaxError, it emits the same program again. A
  trailing `)` is dropped only when doing so turns unparseable code into parseable code.
- **Errors are labelled.** A crashed program comes back as `error: ...` with the traceback,
  and the observation carries "Fix the call and try again". That turns a confidently wrong
  answer into an admission that the program failed. Note that it does not produce a working
  retry: forcing the model to call again after an error made things worse, spending every
  remaining step on variants of the code that had just failed.

Generation stops at `<|tool_call_end|>`, which is the LFM2 chat template's marker. The check
decodes a 24-token window rather than everything generated so far: decoding the whole tail on
every token is quadratic, invisible on a thirty-token reply and real on the `code` agent's
four hundred. A model
that emits a bare `read_file('x')` line instead is also parsed; a model with a different
marker needs `TOOL_CALL_START`/`TOOL_CALL_END` in `agent.py` changed to match.

## Evaluating

```bash
scripts/evaluate.sh                      # every case, into runs/
uv run root-eval --agent python          # one agent
uv run root-eval --min-accuracy 0.8      # non-zero exit on regression
```

`configs/evals.yaml` holds cases of prompt, expected tool, expected argument substring, and
expected answer substrings. `expect_tool: none` asserts the agent answered without a tool.
Tool choice and answer content are scored separately, because at this size they fail
independently: the model picks the right tool and then misreads its output.

Cases that search or read files point at `tests/fixtures/workspace` through a per-case
`workspace:` key, so their answers do not shift when this repository changes.

Current score on `LFM2.5-350M`: **97/114 cases, tool choice 97%, answer content 88%**. See
[Eval findings](#eval-findings) for every model and what the numbers showed, and
[Instruction following](#instruction-following) for the two published benchmarks, which score
the model rather than the agents.

## Instruction following

`root-eval` scores the agents this repository ships. Two published benchmarks score the
model itself, on plain instruction following with no tools involved:

```bash
scripts/instruct_eval.sh                          # every sub-1B model, all GPUs listed in GPUS
GPUS=1,2,3 scripts/instruct_eval.sh lfm2-350m     # one model, three cards
uv run root-instruct --limit 8 --skip-judge       # a smoke run with no judge
```

[IFEval](https://huggingface.co/datasets/google/IFEval) (541 prompts, 25 constraint types)
asks for things a program can check: no commas, at least 300 words, end with this exact
phrase, answer in Kannada. `ifeval.py` holds one verifier per constraint and reports the
four numbers the benchmark is published with - prompt level and instruction level, each
strict and loose. Loose forgives a "Sure, here it is" preamble, a trailing offer of more
help, and emphasis wrapped round the whole answer, since a model that followed the
instruction and then framed it has still followed the instruction.

[InFoBench](https://huggingface.co/datasets/kqsong/InFoBench) (500 instructions decomposed
into 2,250 yes/no requirements) cannot be checked by a program, so a judge model answers
each requirement and the score is the fraction answered yes. The judge never generates more
than one token: the verdict is read from the probability it puts on `Yes` against the
probability it puts on `No`, which both batches into a single forward pass and avoids
parsing an answer out of a model that opens with "Based on the".

Both passes are checkpointed a case at a time, so a run that dies resumes without
regenerating, and `--report-only` rescores what is on disk after a verifier changes. That
separation is what made validating the verifiers cheap: they are written from the papers'
described semantics rather than vendored from either project, then checked against Google's
released implementation on 5,004 real verdicts. **24 of the 25 verifiers agree exactly, and
the whole set agrees on 99.86%.** Three disagreements found that way were bugs here and are
fixed: the two case checks also require the response to be English, and `forbidden_words`
matches whole words, so forbidding "can" does not fail a response for "cannot".

The one remaining divergence is deliberate. `keywords:letter_frequency` accepts only a-z in
the reference, which substitutes a *random* letter for anything else - making its verdict on
the two prompts that ask about `!` and `#` arbitrary and unrepeatable. root counts the
character the prompt asked for.

### Results

Every sub-1B model in `models.yaml`, greedy, 1024 new tokens, one run. 6,246 responses,
5,004 IFEval verdicts and 3,000 judged instructions covering 13,500 decomposed requirements.

| Model | IFEval prompt-strict | prompt-loose | inst-strict | InFoBench DRFR | easy | hard |
| --- | --- | --- | --- | --- | --- | --- |
| `LiquidAI/LFM2.5-350M` (default) | **71.2%** | 73.6% | **80.2%** | 44.3% | 47.8% | 42.7% |
| `LiquidAI/LFM2.5-230M` | 66.2% | 69.3% | 75.8% | 38.8% | 42.5% | 37.1% |
| `Qwen/Qwen3-0.6B` | 57.5% | 61.9% | 67.7% | **46.7%** | 50.3% | 45.1% |
| `Qwen/Qwen3.5-0.8B` | 52.3% | 56.4% | 63.1% | 44.5% | 50.0% | 42.1% |
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 26.4% | 28.3% | 37.6% | 26.2% | 33.0% | 23.2% |
| `Qwen/Qwen2.5-Coder-0.5B` (base) | 15.9% | 19.2% | 28.5% | 16.3% | 15.5% | 16.7% |

The two benchmarks disagree about who wins, and that is the interesting part. LFM2.5-350M
leads IFEval by 14 points over Qwen3-0.6B and *loses* InFoBench to it. IFEval asks whether
the output has the demanded shape; InFoBench asks whether it does what was wanted. A model
tuned hard on format compliance tops the first and not the second, so quoting either number
alone will mislead you about the same pair of models.

The rest reads as expected. Loose beats strict by two to four points everywhere, which is the
cost of a "Sure, here it is" preamble. InFoBench's hard split is five to ten points below its
easy split for every model. The two code-tuned entries are far below the general models on
both, and the base checkpoint sits near the floor: with no instruction tuning it repeats a
single phrase until the token budget runs out, so it is a floor reading rather than a score.
`gemma-3-270m` and `gemma-3-1b` are gated and could not be downloaded; `k2-horizon-09b` is
1.08B parameters despite its name, so it is over the ceiling for this table.

DRFR here is not comparable to the paper's GPT-4 figures - see the judge caveats below - but
it is comparable across these six, which were judged by the same model in the same run.

Two departures from the InFoBench protocol are worth knowing before comparing against
published DRFR: the judge is a local open model rather than GPT-4, and each requirement is
judged on its own rather than as a turn in one growing conversation. Sentence counting goes
through nltk's punkt and language identification through langdetect, which is what the
reference IFEval harness uses; `uv sync --extra eval` installs both.

## What to expect at 350M

Single-hop tool use is reliable once the right agent is chosen. Reading a long file and
reasoning over it is not:
`read_file` truncates at 800 characters, and the model often summarizes the top of a file
instead of answering. Picking among many tool results is likewise weak. Raise
`MAX_TOOL_OUTPUT_CHARS` and move to a 1.2B model before trusting it on real documents.

## Tests

```bash
uv run pytest
```

321 tests, none of which need weights or a GPU: the agent loop runs against a scripted model,
and the formats, call grammars, router, engine selection, decoding config, stream gate,
terminal commands and trace rendering are tested as plain functions.
