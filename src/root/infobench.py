"""InFoBench: instruction following scored one requirement at a time.

Each instruction is decomposed by hand into yes/no questions - "is the text a
post title?", "is it suitable for the given input?" - and the metric is the
Decomposed Requirements Following Ratio: the fraction of those questions a
judge answers yes for. Unlike IFEval nothing here is checkable by a program,
so the judge is a model.

Dataset and metric from Qin et al., "InFoBench: Evaluating Instruction
Following Ability in Large Language Models" (arXiv:2401.03601), `kqsong/InFoBench`
on the Hub. Two deliberate departures from the paper's protocol, both of which
move the absolute numbers and are reported alongside them:

  - The judge is a local open model reading P(Yes) off its own logits, not
    GPT-4. Any DRFR here is comparable across the models scored in the same run
    and not to the published figures.
  - Each question is judged on its own rather than as a turn in one growing
    conversation, so a later question cannot be swayed by an earlier verdict.
    That costs the paper's ordering effect and buys a batched forward pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The judge's standing instructions. Paraphrased from the rubric in the paper:
# yes only for complete compliance, and no when the text gives nothing to judge.
JUDGE_SYSTEM = (
    "You grade whether a piece of generated text satisfies one specific "
    "requirement. Answer with a single word, Yes or No.\n"
    "Answer Yes only if the text fully satisfies the requirement. A partial or "
    "near miss is No: if the requirement is that every sentence does something "
    "and one sentence does not, the answer is No.\n"
    "Answer No when the text does not meet the requirement, and also when it "
    "contains nothing that bears on the question at all."
)


@dataclass(frozen=True, slots=True)
class Row:
    id: str
    instruction: str
    input: str
    category: str
    subset: str
    questions: tuple[str, ...]
    labels: tuple[tuple[str, ...], ...]

    @property
    def prompt(self) -> str:
        """The instruction as a model is asked it.

        Half the set carries an input the instruction refers to; the other half
        is the instruction alone.
        """
        return f"{self.instruction}\n\n{self.input}" if self.input.strip() else self.instruction


@dataclass(frozen=True, slots=True)
class RowScore:
    id: str
    subset: str
    labels: tuple[tuple[str, ...], ...]
    followed: tuple[bool, ...]
    # P(Yes) normalised against P(No), kept so a borderline verdict is visible
    # rather than collapsed into a bit.
    confidence: tuple[float, ...]


def load_rows(path: Path) -> list[Row]:
    """Read the released file, which is JSON lines despite its .json name."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        rows.append(
            Row(
                id=entry["id"],
                instruction=entry["instruction"],
                input=entry.get("input") or "",
                category=entry.get("category") or "",
                subset=entry.get("subset") or "",
                questions=tuple(entry["decomposed_questions"]),
                labels=tuple(tuple(label) for label in entry.get("question_label") or ()),
            )
        )
    return rows


def judge_messages(row: Row, response: str, question: str) -> list[dict[str, str]]:
    parts = []
    if row.input.strip():
        parts.append(f"Input:\n{row.input}")
    parts.append(f"Generated text:\n{response}")
    parts.append(f"Requirement: {question}")
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def drfr(scores: list[RowScore]) -> dict[str, float]:
    """The ratio over every question in the run.

    Averaged over questions rather than over instructions, so an instruction
    decomposed into fifteen requirements weighs more than one decomposed into
    two - which is the paper's definition and the reason the metric is a ratio
    and not an accuracy.
    """
    verdicts = [followed for score in scores for followed in score.followed]
    if not verdicts:
        return {}
    return {
        "drfr": sum(verdicts) / len(verdicts),
        "instructions": len(scores),
        "questions": len(verdicts),
        "fully_followed": sum(all(score.followed) for score in scores) / len(scores),
    }


def by_subset(scores: list[RowScore]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RowScore]] = {}
    for score in scores:
        grouped.setdefault(score.subset or "unlabelled", []).append(score)
    return {subset: drfr(rows) for subset, rows in sorted(grouped.items())}


def by_label(scores: list[RowScore]) -> dict[str, dict[str, float]]:
    """Ratio per requirement category, for seeing what kind of rule breaks.

    A question carrying two labels counts under both.
    """
    tally: dict[str, list[bool]] = {}
    for score in scores:
        for labels, followed in zip(score.labels, score.followed, strict=False):
            for label in labels or ("unlabelled",):
                tally.setdefault(label, []).append(followed)
    return {
        label: {"drfr": sum(hits) / len(hits), "questions": len(hits)}
        for label, hits in sorted(tally.items())
    }
