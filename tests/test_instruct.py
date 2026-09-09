import json

import pytest

from root.config import MAX_PARAMETERS, ModelChoice
from root.instruct import (
    IFEVAL,
    INFOBENCH,
    SOURCES,
    _single_token_ids,
    append,
    prompts_for,
    read_checkpoint,
    sub_billion,
    write_json,
)


class OneWordTokenizer:
    """Encodes only the forms it was given as single tokens.

    A real tokeniser splits " YES" into two pieces on some vocabularies and one
    on others, which is the whole reason the judge collects ids rather than
    assuming a spelling.
    """

    def __init__(self, single):
        self.single = dict(single)

    def encode(self, text, add_special_tokens=True):
        if text in self.single:
            return [self.single[text]]
        return [0] * (len(text) + 1)


def test_the_judge_collects_every_single_token_spelling_of_a_label():
    tokenizer = OneWordTokenizer({"Yes": 10, " Yes": 11, "yes": 12})
    assert _single_token_ids(tokenizer, ("Yes", "yes", "YES")) == [10, 11, 12]


def test_a_label_with_no_single_token_form_is_an_error():
    """Silently returning no ids would make every verdict a coin flip on 1e-9."""
    with pytest.raises(ValueError, match="single token"):
        _single_token_ids(OneWordTokenizer({}), ("Yes",))


def test_models_over_the_in_process_ceiling_are_left_out(monkeypatch):
    counts = {"vendor/small": 500_000_000, "vendor/huge": 30_000_000_000}
    monkeypatch.setattr("root.instruct.hub_parameters", counts.get)
    models = {
        "small": ModelChoice("small", "vendor/small"),
        "huge": ModelChoice("huge", "vendor/huge"),
    }

    keep, skipped = sub_billion(models)

    assert keep == ["small"]
    assert "huge" in skipped and "30.0B" in skipped["huge"]


def test_a_model_the_hub_gives_no_count_for_is_kept(monkeypatch):
    """Dropping it would mean dropping every model with incomplete metadata."""
    monkeypatch.setattr("root.instruct.hub_parameters", lambda _id: None)
    keep, skipped = sub_billion({"mystery": ModelChoice("mystery", "vendor/mystery")})
    assert keep == ["mystery"]
    assert not skipped


def test_the_ceiling_used_is_the_one_the_rest_of_root_enforces(monkeypatch):
    monkeypatch.setattr("root.instruct.hub_parameters", lambda _id: MAX_PARAMETERS + 1)
    keep, _ = sub_billion({"over": ModelChoice("over", "vendor/over")})
    assert keep == []


def test_a_checkpoint_reads_back_as_id_to_response(tmp_path):
    path = tmp_path / "ifeval_model.jsonl"
    append(path, [{"id": "1", "prompt": "p", "response": "a"}])
    append(path, [{"id": "2", "prompt": "q", "response": "b"}])
    assert read_checkpoint(path) == {"1": "a", "2": "b"}


def test_a_later_line_for_the_same_id_wins(tmp_path):
    """A run killed mid-batch can leave a case written twice; the rerun's answer
    is the one that matches the model and settings now in play."""
    path = tmp_path / "ifeval_model.jsonl"
    append(path, [{"id": "1", "prompt": "p", "response": "first"}])
    append(path, [{"id": "1", "prompt": "p", "response": "second"}])
    assert read_checkpoint(path) == {"1": "second"}


def test_a_missing_checkpoint_is_an_empty_run_not_an_error(tmp_path):
    assert read_checkpoint(tmp_path / "nothing.jsonl") == {}


def test_the_report_is_replaced_in_one_step(tmp_path):
    """Several models write to one output directory, so a half-written report
    would be read by the next process as the run's result."""
    path = tmp_path / "report.json"
    write_json(path, {"models": {"a": 1}})
    write_json(path, {"models": {"a": 1, "b": 2}})
    assert json.loads(path.read_text())["models"] == {"a": 1, "b": 2}
    assert list(tmp_path.iterdir()) == [path]


def test_limit_takes_the_first_cases_of_either_benchmark(tmp_path):
    ifeval = tmp_path / "ifeval.jsonl"
    ifeval.write_text(
        "\n".join(
            json.dumps({"key": key, "prompt": f"p{key}", "instruction_id_list": [], "kwargs": []})
            for key in range(5)
        )
    )
    assert prompts_for(IFEVAL, ifeval, 2) == [("0", "p0"), ("1", "p1")]
    assert len(prompts_for(IFEVAL, ifeval, None)) == 5


def test_infobench_prompts_carry_the_input_the_instruction_needs(tmp_path):
    path = tmp_path / "InfoBench.json"
    path.write_text(
        json.dumps(
            {
                "id": "task_1",
                "input": "Some source text.",
                "category": "Test",
                "instruction": "Summarise it.",
                "decomposed_questions": ["Is it a summary?"],
                "subset": "Easy_set",
                "question_label": [["Content"]],
            }
        )
    )
    [(case_id, prompt)] = prompts_for(INFOBENCH, path, None)
    assert case_id == "task_1"
    assert "Some source text." in prompt and "Summarise it." in prompt


@pytest.mark.parametrize("benchmark", [IFEVAL, INFOBENCH])
def test_each_benchmark_names_the_file_it_downloads(benchmark):
    url, name = SOURCES[benchmark]
    assert url.startswith("https://huggingface.co/datasets/")
    assert url.endswith(name)


class ScriptedJudge:
    """Hands back canned confidences so the checkpointing can be tested."""

    def __init__(self, confidences):
        self.remaining = list(confidences)
        self.batches = []

    def verdicts(self, conversations):
        self.batches.append(len(conversations))
        taken = self.remaining[: len(conversations)]
        self.remaining = self.remaining[len(conversations) :]
        return taken


def test_an_instruction_is_written_only_once_every_question_is_judged(tmp_path):
    """A batch boundary falls in the middle of an instruction's questions, and a
    row written halfway would be read back as a finished row with fewer
    requirements than it has."""
    from root.infobench import Row
    from root.instruct import judge_responses

    rows = [
        Row("a", "Do it.", "", "c", "Easy_set", ("q1", "q2", "q3"), (("Content",),) * 3),
        Row("b", "Do it too.", "", "c", "Easy_set", ("q1",), (("Content",),)),
    ]
    judge = ScriptedJudge([0.9, 0.9, 0.1, 0.8])
    output = tmp_path / "judged.jsonl"

    judge_responses(judge, "m", rows, {"a": "answer a", "b": "answer b"}, output, batch_size=2)

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert {entry["id"] for entry in written} == {"a", "b"}
    by_id = {entry["id"]: entry for entry in written}
    assert by_id["a"]["followed"] == [True, True, False]
    assert by_id["b"]["followed"] == [True]


def test_an_already_judged_instruction_is_not_judged_again(tmp_path):
    from root.infobench import Row
    from root.instruct import judge_responses

    rows = [Row("a", "Do it.", "", "c", "Easy_set", ("q1",), (("Content",),))]
    output = tmp_path / "judged.jsonl"
    append(output, [{"id": "a", "confidence": [0.9], "followed": [True]}])
    judge = ScriptedJudge([])

    judge_responses(judge, "m", rows, {"a": "answer"}, output, batch_size=2)

    assert judge.batches == []


def test_an_instruction_with_no_saved_answer_is_not_judged(tmp_path):
    """A model whose generation failed for a case has nothing to grade, and
    grading an empty string would score it as a followed requirement."""
    from root.infobench import Row
    from root.instruct import judge_responses

    rows = [Row("a", "Do it.", "", "c", "Easy_set", ("q1",), (("Content",),))]
    judge = ScriptedJudge([])
    judge_responses(judge, "m", rows, {}, tmp_path / "judged.jsonl", batch_size=2)
    assert judge.batches == []


def test_a_subset_with_no_rows_prints_a_dash_rather_than_raising():
    """A --limit run can cover only one of the two subsets, and formatting a
    missing ratio as a percentage is a crash at the end of a finished run."""
    from root.instruct import _percent

    assert _percent(None) == "-"
    assert _percent(0.5) == "50.0%"


class OverloadedJudge:
    """Raises out-of-memory until the batch is small enough."""

    def __init__(self, survivable):
        self.survivable = survivable
        self.batches = []

    def verdicts(self, conversations):
        import torch

        self.batches.append(len(conversations))
        if len(conversations) > self.survivable:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return [0.9] * len(conversations)


def test_the_judge_halves_its_batch_rather_than_losing_the_run(tmp_path):
    """A base model that never stops talking makes judge prompts several times
    longer, and attention over them is quadratic; the batch size that suited
    the previous model is then far too large."""
    from root.infobench import Row
    from root.instruct import judge_responses

    rows = [Row(str(i), "Do it.", "", "c", "Easy_set", ("q1",), (("Content",),)) for i in range(8)]
    judge = OverloadedJudge(survivable=2)
    output = tmp_path / "judged.jsonl"

    judge_responses(judge, "m", rows, {row.id: "answer" for row in rows}, output, batch_size=8)

    assert judge.batches[0] == 8
    assert max(judge.batches[-4:]) <= 2
    assert len({json.loads(line)["id"] for line in output.read_text().splitlines()}) == 8


def test_a_single_question_that_will_not_fit_is_raised_not_swallowed(tmp_path):
    """Halving to one and still failing is a real problem with the judge or the
    card, and quietly scoring zero questions would look like a finished run."""
    import torch

    from root.infobench import Row
    from root.instruct import judge_responses

    rows = [Row("a", "Do it.", "", "c", "Easy_set", ("q1",), (("Content",),))]
    with pytest.raises(torch.OutOfMemoryError):
        judge_responses(
            OverloadedJudge(survivable=0),
            "m",
            rows,
            {"a": "answer"},
            tmp_path / "judged.jsonl",
            batch_size=2,
        )
