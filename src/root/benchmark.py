from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Decoding, load_models, resolve_model
from .engines import AUTO, ENGINES, TRANSFORMERS, EngineUnavailable, load_engine

PROMPT = "Write a short paragraph about why small language models are useful on a laptop."


@dataclass(frozen=True, slots=True)
class Measurement:
    engine: str
    model: str
    served_as: str
    runs: int
    concurrency: int
    tokens: int
    seconds: float
    tokens_per_second: float
    first_token_seconds: float | None

    def line(self) -> str:
        first = f"{self.first_token_seconds:.2f}s" if self.first_token_seconds else "-"
        return (
            f"{self.engine:14s} c={self.concurrency:<2d} {self.tokens_per_second:7.1f} tok/s   "
            f"first token {first:>6s}   {self.tokens:5d} tokens in {self.seconds:.1f}s"
        )


def _one_run(model, decoding: Decoding, prompt: str) -> tuple[int, float, float | None]:
    started = time.monotonic()
    marks: list[float] = []

    def mark(_text: str) -> None:
        if not marks:
            marks.append(time.monotonic() - started)

    result = model.generate([{"role": "user", "content": prompt}], decoding=decoding, on_token=mark)
    return result.generated_tokens, result.seconds, marks[0] if marks else None


def measure(
    model, engine: str, tokens: int, runs: int, concurrency: int = 1, warmup: bool = True
) -> Measurement:
    """Time generation only.

    Sampling is off so the runs are comparable, and the warm-up is discarded:
    the first call pays for lazy kernel compilation and for the model being
    paged in. Above concurrency 1 the requests overlap and throughput is
    measured against the wall clock, which is the workload a serving engine is
    built for and a single stream never shows.
    """
    decoding = Decoding(temperature=0.0, max_new_tokens=tokens)
    if warmup:
        model.generate(
            [{"role": "user", "content": PROMPT}],
            decoding=Decoding(temperature=0.0, max_new_tokens=8),
        )

    started = time.monotonic()
    if concurrency == 1:
        outcomes = [_one_run(model, decoding, PROMPT) for _ in range(runs)]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(
                pool.map(lambda _: _one_run(model, decoding, PROMPT), range(runs * concurrency))
            )
    wall = time.monotonic() - started

    generated = sum(count for count, _, _ in outcomes)
    first_token = [mark for _, _, mark in outcomes if mark is not None]
    # Serial runs bill only generation time; overlapping ones have to use the
    # wall clock, since their generation times add up to more than elapsed.
    elapsed = sum(seconds for _, seconds, _ in outcomes) if concurrency == 1 else wall

    return Measurement(
        engine=engine,
        model=model.model_id,
        served_as=getattr(model, "served_as", model.model_id),
        runs=runs,
        concurrency=concurrency,
        tokens=generated,
        seconds=elapsed,
        tokens_per_second=generated / elapsed if elapsed else 0.0,
        first_token_seconds=statistics.mean(first_token) if first_token else None,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="root-bench",
        description="Measure generation throughput for each inference engine.",
    )
    parser.add_argument("--model", default="qwen3-06b", help="alias or Hugging Face id")
    parser.add_argument(
        "--engines",
        default=",".join(e for e in ENGINES if e != AUTO),
        help="comma separated",
    )
    parser.add_argument("--tokens", type=int, default=128, help="tokens to generate per run")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="requests in flight at once; above 1 measures batched serving",
    )
    parser.add_argument("--device", help="for the transformers engine")
    parser.add_argument("--output", type=Path, help="write measurements as JSON here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    choice = resolve_model(args.model, load_models())
    wanted = [name.strip() for name in args.engines.split(",") if name.strip()]

    measurements = []
    for engine in wanted:
        if engine not in ENGINES:
            print(f"unknown engine {engine!r}; try {list(ENGINES)}", file=sys.stderr)
            return 1
        if engine == TRANSFORMERS and args.concurrency > 1:
            # One in-process model, one set of Metal command buffers: calling
            # generate from several threads aborts the process rather than
            # batching. The served engines are the ones that queue requests.
            print(f"{engine:14s} skipped: the in-process engine cannot serve concurrent requests")
            continue
        try:
            if engine == TRANSFORMERS:
                model = load_engine(
                    choice.id,
                    engine,
                    device=args.device,
                    trust_remote_code=choice.trust_remote_code,
                )
            else:
                model = load_engine(choice.id, engine)
            result = measure(model, engine, args.tokens, args.runs, args.concurrency)
            model.unload()
        except EngineUnavailable as exc:
            print(f"{engine:14s} unavailable: {exc}")
            continue
        measurements.append(result)
        print(result.line())

    if measurements and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([asdict(m) for m in measurements], indent=2), encoding="utf-8"
        )

    if measurements:
        fastest = max(measurements, key=lambda m: m.tokens_per_second)
        print(f"\nfastest: {fastest.engine} at {fastest.tokens_per_second:.1f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
