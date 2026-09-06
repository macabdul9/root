# root

Small tool-using agents that run entirely on a local sub-billion-parameter open-weight
model. Lowercase, because it is smol.

The default model is `LiquidAI/LFM2.5-350M`; any instruct model whose chat template accepts
a `tools` argument works via `--model`.

## Setup

```bash
scripts/setup.sh
source .venv/bin/activate
```

## The terminal

`root` with no prompt opens an interactive session. The model loads once and stays loaded,
which is the difference between 4 seconds a question and 4 seconds plus 15 of startup.

```
$ root
root · LFM2.5-350M · chat · /help for commands, ctrl-d to leave
chat> /agent calc
agent: calc
calc> What is 17 times 23, plus 4?
  calculator('17 * 23 + 4') -> 395
The result of 17 times 23 plus 4 is 395.
[0.6s]
calc> Now subtract 95 from that.
  calculator('17 * 23 - 95') -> 296
...
```

| Command | Effect |
| --- | --- |
| `/agents` | List configured agents, marking the current one |
| `/agent NAME` | Switch agent, clearing the conversation |
| `/tools` | Show what the current agent may call |
| `/trace [LEVEL]` | Show or set the trace level: `off`, `on`, `full` |
| `/last` | Replay the previous turn in full detail |
| `/workspace [PATH]` | Show or change where the file tools look |
| `/new` | Forget the conversation so far |
| `/help`, `/quit` | The obvious |

The last four exchanges carry into the next prompt; the tool calls behind them do not, since
replaying old observations into a 350M context invites it to answer from a stale one. Line
editing and history come from readline, kept in `~/.root_history`.

## The trace

Every step prints as it finishes, so a slow call is visible rather than a hang, and a wrong
answer can be traced to the step that produced it. `--trace` sets the level for one-shot
runs, `/trace` inside a session.

At `on`, the default, each tool call shows its argument, its result, and what the step cost:

```
python> what is 2+2
  ● run_python('2+2')
    ⎿ 4
      0.3s · 12 tok · tool 0.0s · forced
The sum of 2 and 2 is 4.
[0.6s · 25 tok · 1 call]
```

`forced` means the turn was opened with the tool-call marker rather than chosen freely. The
closing line totals the turn, and says `agent has no tools` when there were none to call:

```
chat> [2,3,41,0,19,-10] sort this number
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
| `chat` | none | Short direct answers |
| `calc` | calculator | Arithmetic word problems |
| `python` | run_python | Anything a short program answers better than the model |
| `files` | read_file, list_files | Questions about files in the workspace |
| `search` | grep, read_file | Finding where something is defined or used |
| `extract` | none | Text to a fixed JSON object |

## Tools

| Tool | Argument | Returns |
| --- | --- | --- |
| `calculator` | expression | The number, via an AST walk (no `eval`) |
| `run_python` | code | Whatever the program printed |
| `read_file` | path | File contents, truncated at 800 characters |
| `list_files` | glob | Matching workspace paths |
| `grep` | pattern or regex | `path:line: text` for each match, capped at 30 |
| `file_info` | path | Line count, byte size, modification time |
| `json_get` | `file.json:dotted.path` | One value out of a JSON file |
| `today` | empty or `+3 days` | Current date and time |

File tools resolve paths inside the workspace and refuse anything that escapes it.

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
scripts/evaluate.sh                      # every case
scripts/evaluate.sh --agent python       # one agent
scripts/evaluate.sh --min-accuracy 0.8   # non-zero exit on regression
```

`configs/evals.yaml` holds cases of prompt, expected tool, expected argument substring, and
expected answer substrings. `expect_tool: none` asserts the agent answered without a tool.
Tool choice and answer content are scored separately, because at this size they fail
independently: the model picks the right tool and then misreads its output.

Cases that search or read files point at `tests/fixtures/workspace` through a per-case
`workspace:` key, so their answers do not shift when this repository changes.

Current score on `LFM2.5-350M`: **10/11 cases, tool choice 100%, answer content 91%**. The
failing case asks for the number of days between two dates; the model writes broken
`datetime` code, and rather than fixing it, reports that the program failed.

## What to expect at 350M

Single-hop tool use is reliable. Reading a long file and reasoning over it is not:
`read_file` truncates at 800 characters, and the model often summarizes the top of a file
instead of answering. Picking among many tool results is likewise weak. Raise
`MAX_TOOL_OUTPUT_CHARS` and move to a 1.2B model before trusting it on real documents.

## Tests

```bash
pytest
```

65 tests, none of which need weights or a GPU: the agent loop runs against a scripted model,
and the terminal commands and trace rendering are tested as plain functions.
