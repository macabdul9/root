"""Run IFEval and InFoBench over the models root can load in process.

Generation and scoring are separate passes over the same checkpoint files, so a
run that dies halfway - a node going away, a model that will not download -
resumes without regenerating anything, and a change to a verifier is rescored
without touching a GPU.

One model is loaded at a time and answers both benchmarks before it is dropped;
the InFoBench judge is loaded once at the end and grades every model's saved
responses, because it is the largest set of weights in the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from . import ifeval, infobench
from .backend import LocalModel
from .config import MAX_PARAMETERS, Decoding, hub_parameters, load_models, resolve_model

logger = logging.getLogger(__name__)

IFEVAL = "ifeval"
INFOBENCH = "infobench"
BENCHMARKS = (IFEVAL, INFOBENCH)

SOURCES = {
    IFEVAL: (
        "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl",
        "ifeval_input_data.jsonl",
    ),
    INFOBENCH: (
        "https://huggingface.co/datasets/kqsong/InFoBench/resolve/main/InfoBench.json",
        "InfoBench.json",
    ),
}

# IFEval asks for 300-word summaries and 10-paragraph essays, so a budget that
# suits a chat turn truncates the answer and scores it as a failed instruction.
DEFAULT_MAX_NEW_TOKENS = 1024
JUDGE_MODEL = "Qwen/Qwen3.8-27B"

YES_FORMS = ("Yes", "yes", "YES")
NO_FORMS = ("No", "no", "NO")


def fetch(benchmark: str, data_dir: Path) -> Path:
    """The dataset file, downloaded on first use and reused after.

    Kept outside the repository, beside the model weights it is used with, so
    the checkout stays free of a hundred kilobytes of someone else's data.
    """
    url, name = SOURCES[benchmark]
    target = data_dir / name
    if target.is_file() and target.stat().st_size:
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def prompts_for(benchmark: str, path: Path, limit: int | None) -> list[tuple[str, str]]:
    """(id, prompt) for every case, in file order."""
    if benchmark == IFEVAL:
        pairs = [(str(row.key), row.prompt) for row in ifeval.load_rows(path)]
    else:
        pairs = [(row.id, row.prompt) for row in infobench.load_rows(path)]
    return pairs[:limit] if limit else pairs


def read_checkpoint(path: Path) -> dict[str, str]:
    """Responses already written, keyed by case id.

    A later line for the same id wins, which is what makes a rerun after a
    partial batch idempotent rather than duplicated.
    """
    if not path.is_file():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            done[entry["id"]] = entry["response"]
    return done


def append(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate(
    model: LocalModel,
    alias: str,
    benchmark: str,
    pairs: list[tuple[str, str]],
    output: Path,
    decoding: Decoding,
    batch_size: int,
) -> None:
    done = read_checkpoint(output)
    todo = [(case_id, prompt) for case_id, prompt in pairs if case_id not in done]
    if not todo:
        print(f"  {alias} {benchmark}: {len(done)} already done", file=sys.stderr)
        return

    print(
        f"  {alias} {benchmark}: {len(todo)} to generate ({len(done)} cached)",
        file=sys.stderr,
        flush=True,
    )
    started = time.monotonic()
    for index in range(0, len(todo), batch_size):
        batch = todo[index : index + batch_size]
        answers = model.complete_batch([prompt for _, prompt in batch], decoding)
        append(
            output,
            [
                {"id": case_id, "prompt": prompt, "response": answer}
                for (case_id, prompt), answer in zip(batch, answers, strict=True)
            ],
        )
        elapsed = time.monotonic() - started
        finished = index + len(batch)
        print(
            f"    {finished}/{len(todo)} in {elapsed:.0f}s "
            f"({finished / max(elapsed, 1e-9):.2f} cases/s)",
            file=sys.stderr,
            flush=True,
        )


@dataclass(slots=True)
class Judge:
    """A yes/no grader that never generates more than one token.

    The verdict is read from the probability the model puts on Yes against the
    probability it puts on No, which is both cheaper than sampling an answer and
    steadier: a model asked to reply in one word will still sometimes open with
    "Based on", and a parser then has to guess what it meant.
    """

    model_id: str
    tokenizer: object
    model: object
    device: str
    yes_ids: list[int]
    no_ids: list[int]

    @classmethod
    def load(cls, model_id: str, device: str | None = None) -> Judge:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .backend import pick_device

        device = device or pick_device()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
        model.eval()
        return cls(
            model_id=model_id,
            tokenizer=tokenizer,
            model=model,
            device=device,
            yes_ids=_single_token_ids(tokenizer, YES_FORMS),
            no_ids=_single_token_ids(tokenizer, NO_FORMS),
        )

    def verdicts(self, conversations: list[list[dict[str, str]]]) -> list[float]:
        """P(Yes) against P(No) for each conversation, in [0, 1]."""
        import torch

        rendered = [
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
            for messages in conversations
        ]
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            inputs = self.tokenizer(
                rendered, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = side

        with torch.inference_mode():
            # One token, asked for through generate so transformers keeps logits
            # for just the position that matters instead of the whole sequence.
            output = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        probabilities = output.scores[0].float().softmax(dim=-1)
        yes = probabilities[:, self.yes_ids].sum(dim=-1)
        no = probabilities[:, self.no_ids].sum(dim=-1)
        return (yes / (yes + no).clamp(min=1e-9)).tolist()


def _single_token_ids(tokenizer, words: tuple[str, ...]) -> list[int]:
    """Ids for the surface forms of a label that are one token on their own.

    Both the bare word and the space-prefixed form, because whether a model
    writes " Yes" or "Yes" first depends on its template, not on its verdict.
    """
    ids = set()
    for word in words:
        for form in (word, f" {word}"):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
    if not ids:
        raise ValueError(f"{words[0]!r} is not a single token for this judge")
    return sorted(ids)


def judge_responses(
    judge: Judge,
    alias: str,
    rows: list[infobench.Row],
    responses: dict[str, str],
    output: Path,
    batch_size: int,
) -> None:
    """Grade every decomposed question of every answered instruction."""
    import torch

    already = {entry["id"] for entry in _read_json_lines(output)}
    questions: list[tuple[str, list[dict[str, str]]]] = []
    for row in rows:
        if row.id in already or row.id not in responses:
            continue
        for question in row.questions:
            questions.append((row.id, infobench.judge_messages(row, responses[row.id], question)))
    if not questions:
        print(f"  {alias}: nothing left to judge", file=sys.stderr)
        return

    print(f"  {alias}: judging {len(questions)} questions", file=sys.stderr, flush=True)
    wanted = {row.id: len(row.questions) for row in rows}
    pending: dict[str, list[float]] = {}
    started = time.monotonic()
    index = 0
    while index < len(questions):
        batch = questions[index : index + batch_size]
        try:
            confidences = judge.verdicts([messages for _, messages in batch])
        except torch.OutOfMemoryError:
            # A batch size that suits one model's answers can be far too large
            # for another's: a base model that never stops talking produces
            # judge prompts several times longer, and attention over them is
            # quadratic. Halving and carrying on beats losing the run.
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"    out of memory; batch size now {batch_size}", file=sys.stderr, flush=True)
            torch.cuda.empty_cache()
            continue
        index += len(batch)
        for (row_id, _), confidence in zip(batch, confidences, strict=True):
            pending.setdefault(row_id, []).append(confidence)
        # An instruction is written the moment its last question has a verdict,
        # so a run that dies loses one batch rather than the whole pass, and a
        # resumed run never sees an instruction graded halfway. A row's
        # questions are contiguous, so a full count means a full row.
        finished = [row_id for row_id, values in pending.items() if len(values) == wanted[row_id]]
        append(
            output,
            [
                {
                    "id": row_id,
                    "confidence": pending[row_id],
                    "followed": [value > 0.5 for value in pending[row_id]],
                }
                for row_id in finished
            ],
        )
        for row_id in finished:
            del pending[row_id]
        print(
            f"    {index}/{len(questions)} in {time.monotonic() - started:.0f}s",
            file=sys.stderr,
            flush=True,
        )


def _read_json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def sub_billion(models: dict) -> tuple[list[str], dict[str, str]]:
    """The aliases root can load in process, and why the others were left out.

    The hub's parameter count is the same one `check_size` refuses on, so the
    set here is exactly the set root will run locally. A model the hub gives no
    count for is kept: the alternative is dropping every model with incomplete
    metadata.
    """
    keep, skipped = [], {}
    for alias, choice in models.items():
        total = hub_parameters(choice.id)
        if total is not None and total > MAX_PARAMETERS:
            skipped[alias] = f"{total / 1e9:.1f}B parameters, over the in-process ceiling"
        else:
            keep.append(alias)
    return keep, skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="root-instruct",
        description="Score models on IFEval and InFoBench.",
    )
    parser.add_argument(
        "--models",
        help="comma-separated aliases; every sub-1B entry in models.yaml by default",
    )
    parser.add_argument("--benchmark", choices=[*BENCHMARKS, "both"], default="both")
    parser.add_argument("--limit", type=int, help="only the first N cases, for a smoke run")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 for greedy, which is what makes a rerun reproducible",
    )
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/instruct"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "root" / "benchmarks",
        help="where the downloaded datasets are cached",
    )
    parser.add_argument("--judge", default=JUDGE_MODEL, help="model that grades InFoBench")
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--judge-device")
    parser.add_argument(
        "--skip-judge", action="store_true", help="generate InFoBench answers but do not grade them"
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="grade InFoBench answers already on disk without generating any",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="score what is already on disk without loading a model",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print the aliases this run would cover, then exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    benchmarks = BENCHMARKS if args.benchmark == "both" else (args.benchmark,)
    models = load_models()
    if args.models:
        aliases = [alias.strip() for alias in args.models.split(",") if alias.strip()]
        skipped = {}
    else:
        aliases, skipped = sub_billion(models)
    if args.list_models:
        # The launcher asks for this rather than keeping its own copy of the
        # sub-1B rule, so the two can never disagree.
        print(" ".join(aliases))
        return 0
    for alias, reason in skipped.items():
        print(f"skipping {alias}: {reason}", file=sys.stderr)

    data = {benchmark: fetch(benchmark, args.data_dir) for benchmark in benchmarks}
    cases = {
        benchmark: prompts_for(benchmark, path, args.limit) for benchmark, path in data.items()
    }
    decoding = Decoding(temperature=args.temperature, max_new_tokens=args.max_new_tokens)

    failures: dict[str, str] = {}
    if not (args.report_only or args.judge_only):
        for alias in aliases:
            choice = resolve_model(alias, models)
            print(f"{alias} ({choice.id})", file=sys.stderr, flush=True)
            try:
                model = LocalModel.load(
                    choice.id, device=args.device, trust_remote_code=choice.trust_remote_code
                )
            except Exception as exc:
                # A gated repository or a missing architecture should cost one
                # model, not the rest of the run.
                failures[alias] = f"{type(exc).__name__}: {exc}"
                print(f"  could not load: {failures[alias]}", file=sys.stderr)
                continue
            try:
                for benchmark in benchmarks:
                    generate(
                        model,
                        alias,
                        benchmark,
                        cases[benchmark],
                        args.output_dir / f"{benchmark}_{alias}.jsonl",
                        decoding,
                        args.batch_size,
                    )
            finally:
                model.unload()

    graded = INFOBENCH in benchmarks and not args.skip_judge and not args.report_only
    if graded:
        rows = infobench.load_rows(data[INFOBENCH])
        wanted = {case_id for case_id, _ in cases[INFOBENCH]}
        rows = [row for row in rows if row.id in wanted]
        print(f"judge {args.judge}", file=sys.stderr, flush=True)
        judge = Judge.load(args.judge, device=args.judge_device)
        for alias in aliases:
            responses = read_checkpoint(args.output_dir / f"{INFOBENCH}_{alias}.jsonl")
            if responses:
                judge_responses(
                    judge,
                    alias,
                    rows,
                    responses,
                    args.output_dir / f"{INFOBENCH}_{alias}_judged.jsonl",
                    args.judge_batch_size,
                )

    report = build_report(args, aliases, benchmarks, data, skipped | failures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "report.json", report)
    print_report(report, benchmarks)
    return 0


def write_json(path: Path, payload: dict) -> None:
    """Replace the file in one step.

    The launcher runs a process per model against a shared output directory, so
    two of them can reach this at once; a truncated report is worse than a
    stale one.
    """
    scratch = path.with_suffix(f".{os.getpid()}.tmp")
    scratch.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(scratch, path)


def build_report(
    args: argparse.Namespace,
    aliases: list[str],
    benchmarks: tuple[str, ...],
    data: dict[str, Path],
    skipped: dict[str, str],
) -> dict:
    report: dict = {
        "judge": args.judge if INFOBENCH in benchmarks and not args.skip_judge else None,
        "decoding": {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "skipped": skipped,
        "models": {},
    }
    ifeval_rows = {}
    if IFEVAL in benchmarks:
        ifeval_rows = {str(row.key): row for row in ifeval.load_rows(data[IFEVAL])}
    infobench_rows = {}
    if INFOBENCH in benchmarks:
        infobench_rows = {row.id: row for row in infobench.load_rows(data[INFOBENCH])}

    for alias in aliases:
        entry: dict = {}
        if IFEVAL in benchmarks:
            responses = read_checkpoint(args.output_dir / f"{IFEVAL}_{alias}.jsonl")
            scores = [
                ifeval.score_row(ifeval_rows[case_id], response)
                for case_id, response in responses.items()
                if case_id in ifeval_rows
            ]
            if scores:
                entry[IFEVAL] = ifeval.report(scores) | {
                    "by_instruction": ifeval.by_instruction(scores)
                }
        if INFOBENCH in benchmarks:
            verdicts = _read_json_lines(args.output_dir / f"{INFOBENCH}_{alias}_judged.jsonl")
            scores = [
                infobench.RowScore(
                    id=entry_row["id"],
                    subset=infobench_rows[entry_row["id"]].subset,
                    labels=infobench_rows[entry_row["id"]].labels,
                    followed=tuple(entry_row["followed"]),
                    confidence=tuple(entry_row["confidence"]),
                )
                for entry_row in verdicts
                if entry_row["id"] in infobench_rows
            ]
            if scores:
                entry[INFOBENCH] = (
                    infobench.drfr(scores)
                    | {"by_subset": infobench.by_subset(scores)}
                    | {"by_label": infobench.by_label(scores)}
                )
        if entry:
            report["models"][alias] = entry
    return report


def _percent(value: float | None) -> str:
    """A percentage, or a dash for a subset this run did not cover.

    A --limit run can leave one of the two subsets with no rows at all, and
    formatting None as a percentage raises rather than printing a gap.
    """
    return "-" if value is None else f"{value:.1%}"


def print_report(report: dict, benchmarks: tuple[str, ...]) -> None:
    models = report["models"]
    if not models:
        print("\nnothing scored yet", file=sys.stderr)
        return
    width = max(len(alias) for alias in models)
    scored = {name for entry in models.values() for name in entry}
    if IFEVAL in benchmarks and IFEVAL in scored:
        prompts = next(
            (entry[IFEVAL]["prompts"] for entry in models.values() if IFEVAL in entry), 0
        )
        print(f"\nIFEval  ({prompts} prompts)")
        print(f"  {'':{width}s}  prompt-strict  prompt-loose  inst-strict  inst-loose")
        for alias, entry in models.items():
            scores = entry.get(IFEVAL)
            if scores:
                print(
                    f"  {alias:{width}s}  {scores['prompt_strict']:>12.1%}  "
                    f"{scores['prompt_loose']:>11.1%}  {scores['instruction_strict']:>10.1%}  "
                    f"{scores['instruction_loose']:>9.1%}"
                )
    if INFOBENCH in benchmarks and INFOBENCH in scored:
        print(f"\nInFoBench  (judge: {report['judge']})")
        print(f"  {'':{width}s}  DRFR    easy    hard    all-met")
        for alias, entry in models.items():
            scores = entry.get(INFOBENCH)
            if scores:
                subsets = scores.get("by_subset", {})
                easy = _percent(subsets.get("Easy_set", {}).get("drfr"))
                hard = _percent(subsets.get("Hard_set", {}).get("drfr"))
                print(
                    f"  {alias:{width}s}  {scores['drfr']:>5.1%}  {easy:>6}  {hard:>6}  "
                    f"{scores['fully_followed']:>7.1%}"
                )
    for alias, reason in report["skipped"].items():
        print(f"\nnot scored: {alias} - {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
