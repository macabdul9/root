import json

from root.infobench import (
    JUDGE_SYSTEM,
    Row,
    RowScore,
    by_label,
    by_subset,
    drfr,
    judge_messages,
    load_rows,
)

WITH_INPUT = Row(
    id="task_1",
    instruction="Choose an appealing title for your post.",
    input="Avocados are high in calories.",
    category="Quora",
    subset="Easy_set",
    questions=("Is the text a post title?", "Is it appealing?"),
    labels=(("Format",), ("Content",)),
)
WITHOUT_INPUT = Row(
    id="task_2",
    instruction="Write a limerick about rain.",
    input="",
    category="Poetry",
    subset="Hard_set",
    questions=("Is it a limerick?",),
    labels=(("Format",),),
)


def test_an_instruction_with_an_input_carries_it_into_the_prompt():
    assert WITH_INPUT.instruction in WITH_INPUT.prompt
    assert WITH_INPUT.input in WITH_INPUT.prompt


def test_an_instruction_without_an_input_is_asked_on_its_own():
    """Half the set has no input; appending an empty one leaves a model staring
    at trailing blank lines and inventing something to fill them."""
    assert WITHOUT_INPUT.prompt == WITHOUT_INPUT.instruction


def test_the_judge_is_shown_the_input_the_requirement_refers_to():
    messages = judge_messages(WITH_INPUT, "Why Avocados Are Worth It", WITH_INPUT.questions[0])
    graded = messages[-1]["content"]
    assert messages[0] == {"role": "system", "content": JUDGE_SYSTEM}
    assert WITH_INPUT.input in graded
    assert "Why Avocados Are Worth It" in graded
    assert WITH_INPUT.questions[0] in graded


def test_the_judge_is_not_shown_an_empty_input_block():
    graded = judge_messages(WITHOUT_INPUT, "There once was a cloud", "Is it a limerick?")[-1]
    assert "Input:" not in graded["content"]


def test_the_judge_is_told_a_partial_match_is_a_no():
    """The rubric is the whole difference between DRFR and a vibe check."""
    assert "No" in JUDGE_SYSTEM and "Yes" in JUDGE_SYSTEM


SCORES = [
    RowScore("a", "Easy_set", (("Format",), ("Content",)), (True, True), (0.9, 0.8)),
    RowScore("b", "Easy_set", (("Format",),), (False,), (0.2,)),
    RowScore("c", "Hard_set", (("Content",), ("Content",)), (True, False), (0.7, 0.1)),
]


def test_the_ratio_is_over_questions_not_over_instructions():
    """Five questions, three answered yes. Averaging per instruction first would
    say 0.5; the metric weighs an instruction by how many requirements it has."""
    assert drfr(SCORES)["drfr"] == 3 / 5
    assert drfr(SCORES)["questions"] == 5
    assert drfr(SCORES)["instructions"] == 3


def test_fully_following_an_instruction_is_reported_separately():
    assert drfr(SCORES)["fully_followed"] == 1 / 3


def test_an_empty_run_is_empty_rather_than_dividing_by_zero():
    assert drfr([]) == {}


def test_the_two_subsets_are_scored_apart():
    split = by_subset(SCORES)
    assert split["Easy_set"]["drfr"] == 2 / 3
    assert split["Hard_set"]["drfr"] == 1 / 2


def test_requirements_are_grouped_by_the_kind_of_rule_they_are():
    labels = by_label(SCORES)
    assert labels["Format"]["questions"] == 2
    assert labels["Content"]["drfr"] == 2 / 3


def test_the_released_file_is_read_as_json_lines(tmp_path):
    """It is named .json but holds one object per line, and json.load chokes."""
    path = tmp_path / "InfoBench.json"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"task_{index}",
                    "input": "",
                    "category": "Test",
                    "instruction": "Do the thing.",
                    "decomposed_questions": ["Was it done?"],
                    "subset": "Easy_set",
                    "question_label": [["Content"]],
                }
            )
            for index in range(3)
        )
        + "\n"
    )
    rows = load_rows(path)
    assert [row.id for row in rows] == ["task_0", "task_1", "task_2"]
    assert rows[0].questions == ("Was it done?",)
