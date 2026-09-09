import pytest

from root.ifeval import (
    VERIFIERS,
    Row,
    by_instruction,
    count_sentences,
    count_words,
    follows,
    loose_variants,
    report,
    score_row,
)

# One satisfying and one violating response per instruction, with the argument
# shape the released data actually uses. A verifier that silently accepts
# everything is worse than a missing one, so both directions are asserted.
CASES = [
    (
        "change_case:capital_word_frequency",
        {"capital_frequency": 2, "capital_relation": "at least"},
        "THIS IS shouting.",
        "this is not.",
    ),
    (
        "change_case:english_capital",
        {},
        "THIS WHOLE ANSWER IS WRITTEN IN CAPITAL LETTERS FOR YOU.",
        "this whole answer is written in lower case letters for you.",
    ),
    (
        "change_case:english_lowercase",
        {},
        "this whole answer is written in lower case letters for you.",
        "THIS WHOLE ANSWER IS WRITTEN IN CAPITAL LETTERS FOR YOU.",
    ),
    (
        "combination:repeat_prompt",
        {"prompt_to_repeat": "Write a poem."},
        "Write a poem. Roses are red.",
        "Roses are red.",
    ),
    ("combination:two_responses", {}, "First answer.\n******\nSecond answer.", "Only one answer."),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "Send it to [address] on [date].",
        "Send it to [address].",
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.S."},
        "That is all.\n\nP.S. one more thing",
        "That is all.",
    ),
    ("detectable_format:constrained_response", {}, "My answer is yes.", "Yes, definitely."),
    ("detectable_format:json_format", {}, '{"answer": 1}', "answer = 1"),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "Section", "num_sections": 2},
        "Section 1\nintro\nSection 2\nbody",
        "Section 1\nintro only",
    ),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* one\n* two", "* one"),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 2},
        "*first* and *second*",
        "*first* alone",
    ),
    ("detectable_format:title", {}, "<<A Real Title>>\nBody text.", "A Real Title\nBody text."),
    (
        "keywords:existence",
        {"keywords": ["alpha", "beta"]},
        "alpha and beta both appear",
        "alpha appears alone",
    ),
    ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "all good here", "this is bad"),
    (
        "keywords:frequency",
        {"keyword": "cat", "frequency": 2, "relation": "at least"},
        "cat and another cat",
        "one cat only",
    ),
    (
        "keywords:letter_frequency",
        {"letter": "a", "let_frequency": 3, "let_relation": "at least"},
        "aardvark",
        "the bee",
    ),
    (
        "language:response_language",
        {"language": "de"},
        "Dies ist ein deutscher Satz uber das Wetter in Berlin.",
        "This is an English sentence about the weather in London.",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "jasper"},
        "The first paragraph.\n\nJasper wrote the second.",
        "The first paragraph.\n\nSomeone else wrote the second.",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "First paragraph.\n***\nSecond paragraph.",
        "Only one paragraph.",
    ),
    (
        "length_constraints:number_sentences",
        {"num_sentences": 2, "relation": "at least"},
        "One sentence. And a second.",
        "Only one sentence.",
    ),
    (
        "length_constraints:number_words",
        {"num_words": 3, "relation": "at least"},
        "one two three",
        "one two",
    ),
    ("punctuation:no_comma", {}, "no commas at all here", "yes, there is one"),
    (
        "startend:end_checker",
        {"end_phrase": "Does this make sense?"},
        "Here is the plan. Does this make sense?",
        "Here is the plan.",
    ),
    ("startend:quotation", {}, '"the whole answer is quoted"', "the answer is bare"),
]


def test_the_case_table_covers_every_verifier():
    """A new instruction id with no case here is an untested verifier."""
    assert {instruction_id for instruction_id, *_ in CASES} == set(VERIFIERS)


@pytest.mark.parametrize("instruction_id,arguments,good,bad", CASES)
def test_every_instruction_separates_a_pass_from_a_fail(instruction_id, arguments, good, bad):
    assert follows(good, instruction_id, arguments)
    assert not follows(bad, instruction_id, arguments)


@pytest.mark.parametrize("instruction_id,arguments,good,_bad", CASES)
def test_no_instruction_is_followed_by_an_empty_response(instruction_id, arguments, good, _bad):
    """`no_comma` and `english_lowercase` are both true of the empty string, and
    a model that answered nothing has not followed an instruction."""
    assert not follows("", instruction_id, arguments)
    assert not follows("   \n ", instruction_id, arguments)


def test_a_null_argument_is_not_passed_to_the_verifier():
    """The data carries a null for every argument an instruction does not take,
    and a verifier with no parameters would raise on being handed one."""
    arguments = {"num_words": None, "relation": None, "keywords": None}
    assert follows(
        "THIS IS AN ENGLISH SENTENCE IN CAPITALS.", "change_case:english_capital", arguments
    )


def test_a_case_instruction_also_requires_english():
    """Caps alone cannot tell an English answer from a transliterated one, and
    the reference harness checks both."""
    assert not follows(
        "DIES IST EIN DEUTSCHER SATZ UBER DAS WETTER.", "change_case:english_capital", {}
    )
    assert not follows(
        "dies ist ein deutscher satz uber das wetter in berlin.",
        "change_case:english_lowercase",
        {},
    )


def test_a_response_no_detector_can_read_counts_as_following():
    """A response with no linguistic features tells you about langdetect, not
    about the model, and the reference treats that as followed."""
    assert follows("12345 67890 :::", "language:response_language", {"language": "de"})
    assert follows("12345 67890 :::", "change_case:english_lowercase", {})


def test_a_forbidden_word_is_matched_as_a_whole_word():
    """Forbidding "can" must not fail a response for saying "cannot"."""
    arguments = {"forbidden_words": ["can", "ride"]}
    assert follows("I cannot promise pride in the outcome.", "keywords:forbidden_words", arguments)
    assert not follows("I can promise that.", "keywords:forbidden_words", arguments)


def test_a_required_keyword_is_not_bounded_the_same_way():
    """`keywords:existence` is a bare search in the reference, so a keyword
    inside a longer word still counts as present."""
    assert follows("The adoption process", "keywords:existence", {"keywords": ["adopt"]})


def test_relation_words_are_the_two_the_data_uses():
    at_least = {"num_words": 3, "relation": "at least"}
    less_than = {"num_words": 3, "relation": "less than"}
    assert follows("one two three", "length_constraints:number_words", at_least)
    assert not follows("one two three", "length_constraints:number_words", less_than)
    assert follows("one two", "length_constraints:number_words", less_than)


def test_an_unknown_relation_is_an_error_rather_than_a_silent_fail():
    with pytest.raises(ValueError, match="relation"):
        follows("hello", "length_constraints:number_words", {"num_words": 1, "relation": "about"})


def test_postscript_allows_the_spacing_a_model_writes():
    marker = {"postscript_marker": "P.S."}
    for written in ("P.S. note", "p.s. note", "P. S. note"):
        assert follows(f"The answer.\n\n{written}", "detectable_content:postscript", marker)


def test_the_double_postscript_marker_needs_no_trailing_dot():
    assert follows(
        "Done.\n\nP.P.S I forgot", "detectable_content:postscript", {"postscript_marker": "P.P.S"}
    )


def test_two_responses_must_actually_differ():
    same = "The same thing.\n******\nThe same thing."
    assert not follows(same, "combination:two_responses", {})


def test_bold_text_is_not_counted_as_a_bullet():
    """`**Heading**` at the start of a line is emphasis, not a list item."""
    text = "**Heading**\n* one\n* two"
    assert follows(text, "detectable_format:number_bullet_lists", {"num_bullets": 2})


def test_sentences_are_counted_with_punkt_not_by_splitting_on_dots():
    """An abbreviation is not the end of a sentence; a regexp split says it is,
    and `number_sentences` is a tenth of the benchmark."""
    assert count_sentences("Dr. Smith went home. He slept.") == 2


def test_words_are_runs_of_word_characters():
    assert count_words("well-known state-of-the-art models") == 7


def test_loose_forgives_a_preamble_but_strict_does_not():
    response = "Sure, here is your answer:\nall lower case here"
    assert not follows(response, "change_case:english_lowercase", {})
    assert any(
        follows(variant, "change_case:english_lowercase", {})
        for variant in loose_variants(response)
    )


def test_loose_forgives_wrapping_the_answer_in_emphasis():
    """The quotation check reads the first character, so emphasis around the
    whole answer hides the quotes the instruction asked for."""
    response = '*"the whole answer is quoted"*'
    assert not follows(response, "startend:quotation", {})
    assert any(follows(variant, "startend:quotation", {}) for variant in loose_variants(response))


ROW = Row(
    key=1,
    prompt="Answer in lower case with no commas.",
    instruction_ids=("change_case:english_lowercase", "punctuation:no_comma"),
    kwargs=({}, {}),
)


def test_a_row_scores_each_of_its_instructions():
    score = score_row(ROW, "this answer is written in lower case with no punctuation")
    assert score.strict == (True, True)
    assert score.strict_prompt


def test_one_broken_instruction_fails_the_prompt_but_not_the_other_instruction():
    score = score_row(ROW, "this answer is written in lower case, with a comma")
    assert score.strict == (True, False)
    assert not score.strict_prompt


def test_the_report_separates_prompt_level_from_instruction_level():
    """A prompt with two constraints and one met scores 0 at prompt level and
    0.5 at instruction level, which is why both numbers are published."""
    scores = [score_row(ROW, "this answer is written in lower case, with a comma")]
    numbers = report(scores)
    assert numbers["prompt_strict"] == 0.0
    assert numbers["instruction_strict"] == 0.5
    assert numbers["prompts"] == 1
    assert numbers["instructions"] == 2


def test_the_report_is_empty_rather_than_dividing_by_zero():
    assert report([]) == {}


def test_the_per_instruction_breakdown_names_the_rule_that_broke():
    score = score_row(ROW, "this answer is written in lower case, with a comma")
    breakdown = by_instruction([score])
    assert breakdown["punctuation:no_comma"]["strict"] == 0.0
    assert breakdown["change_case:english_lowercase"]["strict"] == 1.0
