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
checkout resolves to the same versions. Run things through uv rather than activating:

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
| `/agents` | List configured agents, marking the current one |
| `/agent NAME` | Switch agent, clearing the conversation (`auto` to route) |
| `/tools` | Show what the current agent may call |
| `/routes` | Show the rules `auto` uses to pick an agent |
| `/agent`, `/model`, `/trace` | With no argument, list the options and how to set them |
| `/model [NAME]` | List models, or load one by alias or Hugging Face id |
| `/trace [LEVEL]` | Show or set the trace level: `off`, `on`, `full` |
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
* lfm2-350m       LiquidAI/LFM2.5-350M     350M, the default
  lfm2-230m       LiquidAI/LFM2.5-230M     230M, the smallest
  qwen3-06b       Qwen/Qwen3-0.6B          600M, JSON tool calls
  qwen35-08b      Qwen/Qwen3.5-0.8B        800M, XML tool calls
  k2-horizon-09b  IFM/K2-Horizon-0.9B      900M, runs code from its own repo
root> /model qwen3-06b
model: Qwen/Qwen3-0.6B · tool calls: qwen-json
```

## Eval findings

Every model runs the same twelve cases in `configs/evals.yaml`, through the same agents and
tools. Tool choice and answer content are scored separately, because they fail separately.
Reproduce any row with `uv run root-eval --model <alias>`. The cases that touch files use a fixture
workspace shipped with the package, so the suite scores the same from a checkout or from an
installed command in an unrelated directory.

| Model | Cases | Tool choice | Answer content |
| --- | --- | --- | --- |
| `LiquidAI/LFM2.5-230M` | 9/12 | 92% | 75% |
| `LiquidAI/LFM2.5-350M` (default) | 11/12 | 100% | 92% |
| `Qwen/Qwen3-0.6B` | 9/12 | 100% | 75% |
| `Qwen/Qwen3.5-0.8B` | 12/12 | 100% | 100% |
| `IFM/K2-Horizon-0.9B` | 11/12 | 100% | 92% |

What the numbers showed, each of which changed the code:

- **Choosing the tool is easy; reading its output is not.** Four of five models pick the
  right tool every time and lose their points on the answer instead. 350M is where the
  second half starts working, which is why it is the default: 230M is 18 points worse on
  answers for a third less memory.
- **The eval caught a bug in a tool, not just in the models.** Four models failed the same
  case, and the transcripts showed why: they passed `list_files` a directory name rather
  than a glob, and K2-Horizon crashed it outright with `tuple index out of range` from
  `Path.glob(".")`. `list_files` now reads a directory name as its contents and refuses
  patterns it cannot glob. That one fix moved 230M from 7/11 to 8/11 and Qwen3.5 to 11/11.
- **A general agent is worse than a routed specialist.** One agent holding all eight tools
  scores 3/8 with tool choice at 62%, against 10/11 and 100% for specialists. Asking the
  model to route instead scores 0/12, so `route.py` uses regular expressions.
- **Date arithmetic is the last thing to fall over.** Three of five still fail it: they write
  broken `datetime` code and, shown the traceback, report the failure rather than fixing it.
  Forcing a retry made it worse, so the loop reports the failure honestly instead.
- **Small models slip in predictable ways worth repairing.** A missing `print()`, a trailing
  `)`, a tool call with unbalanced quotes. Each was a total failure until it was handled,
  and each is deterministic enough to fix in `tools.py` and `formats.py` rather than by
  prompting.
- **Schema text is load-bearing across tools.** Rewording `list_files`'s description changed
  which path the model passed to `read_file` in a different case: `src/pyproject.toml`
  instead of `pyproject.toml`, one case lost. Tool descriptions are part of the prompt for
  every tool the agent holds, so change one and re-run the evals.

`Qwen/Qwen3.5-0.8B` is the only model to pass every case, at roughly twice the parameters of
the default. `LiquidAI/LFM2.5-350M` stays the default for its size and load time; switch with
`/model qwen35-08b` when accuracy matters more.

`IFM/K2-Horizon-0.9B` ships a custom architecture, so loading it executes Python from its
repository. Its entry carries `trust_remote_code: true` and nothing else does; deleting that
line disables the model rather than silently running the code.

## Tool-call formats

The five models write tool calls four different ways, so `formats.py` holds one `CallFormat`
per family and `detect_format` picks between them by looking for a marker in the model's own
chat template. Reading the template beats a per-model table: a model this project has never
heard of still works if it writes calls like one it has.

| Format | Shape |
| --- | --- |
| `lfm2` | `<\|tool_call_start\|>[calculator(expression='6 * 7')]<\|tool_call_end\|>` |
| `qwen-json` | `<tool_call>{"name": "calculator", "arguments": {...}}</tool_call>` |
| `qwen-xml` | `<tool_call><function=calculator><parameter=expression>6 * 7</parameter>...` |
| `ifm` | `<ifm\|tool_calls><ifm\|tool_call>{"name": ...}</ifm\|tool_call>...` |
| `generic` | A bare `calculator('6 * 7')` line, with no way to force a call |

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
| `files` | read_file, list_files | Questions about files in the workspace |
| `write` | write_file, make_directory, list_files | Creating files and directories |
| `search` | grep, read_file | Finding where something is defined or used |
| `extract` | none | Text to a fixed JSON object |

## Tools

| Tool | Argument | Returns |
| --- | --- | --- |
| `calculator` | expression | The number, via an AST walk (no `eval`) |
| `run_python` | code | Whatever the program printed |
| `read_file` | path | File contents, truncated at 800 characters |
| `write_file` | path, content | Creates or overwrites a workspace file |
| `make_directory` | path | Creates a directory and its parents |
| `list_files` | glob or directory | Matching workspace paths |
| `grep` | pattern or regex | `path:line: text` for each match, capped at 30 |
| `file_info` | path | Line count, byte size, modification time |
| `json_get` | `file.json:dotted.path` | One value out of a JSON file |
| `today` | empty or `+3 days` | Current date and time |

File tools resolve paths inside the workspace and refuse anything that escapes it.
`write_file` overwrites without asking and says which it did, so point a session at a
directory you are willing to have rewritten.

Nearly every tool takes one string argument, which is what these models get right most
reliably. `write_file` takes two, and each call format expresses that natively, so a model
can write `write_file(path='a.txt', content='hi')` or name the arguments in any order.

`run_python` executes model-written code in a child interpreter with a 10 second timeout.
That is isolation, not a sandbox: the program runs as you, in the workspace directory, with
network access. Give the `python` agent a workspace you would hand to a stranger, or drop
the tool from the agent's list.

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

Generation stops at `<|tool_call_end|>`, which is the LFM2 chat template's marker. A model
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

Current score on `LFM2.5-350M`: **11/12 cases, tool choice 100%, answer content 92%**. See
[Eval findings](#eval-findings) for every model and what the numbers showed.

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

176 tests, none of which need weights or a GPU: the agent loop runs against a scripted model,
and the formats, router, terminal commands and trace rendering are tested as plain functions.
