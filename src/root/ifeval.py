"""IFEval: instruction following that can be checked by a program.

Each prompt carries a list of instruction ids and their arguments, and every id
maps to a predicate over the response - "no commas", "at least 300 words", "end
with this exact phrase". Nothing here asks a model for an opinion, which is what
makes the score reproducible.

The verifiers are written from the instruction semantics described in Zhou et
al., "Instruction-Following Evaluation for Large Language Models"
(arXiv:2311.07911) and the argument shapes in the released `google/IFEval` data,
rather than vendored from that project's code. Where a choice is not pinned down
by the paper it is noted at the verifier. Sentence and word tokenisation go
through nltk's punkt, the same tokeniser the reference harness uses, because
counting sentences differently is enough on its own to move the score.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

AT_LEAST = "at least"
LESS_THAN = "less than"

# The three answers the `constrained_response` prompts allow, verbatim.
CONSTRAINED_RESPONSES = ("My answer is yes.", "My answer is no.", "My answer is maybe.")


def _compare(count: int, relation: str, target: int) -> bool:
    if relation == LESS_THAN:
        return count < target
    if relation == AT_LEAST:
        return count >= target
    raise ValueError(f"unknown relation {relation!r}")


@functools.lru_cache(maxsize=1)
def _punkt() -> None:
    """Make sure punkt is on disk before the first tokenise.

    Downloads once per NLTK_DATA directory and is quiet when it is already
    there, so a container that seeded it at build time never reaches the network.
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def count_sentences(text: str) -> int:
    import nltk

    _punkt()
    return len(nltk.tokenize.sent_tokenize(text))


def count_words(text: str) -> int:
    """Words as the reference harness counts them: runs of word characters.

    Deliberately not a punkt tokenise - the reference uses a `\\w+` regexp here,
    and the two disagree on hyphenated words often enough to matter at a 300-word
    threshold.
    """
    return len(re.findall(r"\w+", text))


def count_capitalised_words(text: str) -> int:
    import nltk

    _punkt()
    return sum(1 for word in nltk.tokenize.word_tokenize(text) if word.isupper())


def detect_language(text: str) -> str | None:
    """Two-letter language code, or None when langdetect cannot tell.

    langdetect is what the reference harness uses. Its own default is a random
    seed per process, which makes a borderline response score differently on a
    re-run; pinning the seed costs nothing and makes the number repeatable.

    None is a real answer, not a swallowed error: a response with no
    linguistic features at all - digits, punctuation, a single token - tells
    you about the detector rather than the model, and every language
    instruction here treats that case as followed.
    """
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0
    try:
        return detect(text)
    except LangDetectException:
        return None


def keywords_existence(response: str, keywords: list[str]) -> bool:
    return all(re.search(keyword, response, re.IGNORECASE) for keyword in keywords)


def keywords_frequency(response: str, keyword: str, frequency: int, relation: str) -> bool:
    found = len(re.findall(keyword, response, re.IGNORECASE))
    return _compare(found, relation, frequency)


def keywords_forbidden_words(response: str, forbidden_words: list[str]) -> bool:
    """No forbidden word appears as a word of its own.

    Bounded, unlike `keywords:existence`: forbidding "can" must not fail a
    response for saying "cannot", and forbidding "ride" must not fail one for
    "pride".
    """
    return not any(re.search(rf"\b{word}\b", response, re.IGNORECASE) for word in forbidden_words)


def keywords_letter_frequency(
    response: str, letter: str, let_frequency: int, let_relation: str
) -> bool:
    """How often a character appears, case-insensitively.

    Counts whatever the prompt asked for. The reference harness accepts only
    a-z here and substitutes a *random* letter for anything else, which makes
    its verdict on the two prompts that ask about "!" and "#" arbitrary and
    unrepeatable. Two of 541 prompts, and honouring the request is worth more
    than reproducing a coin toss.
    """
    return _compare(response.lower().count(letter.lower()), let_relation, let_frequency)


def language_response_language(response: str, language: str) -> bool:
    detected = detect_language(response)
    return detected is None or detected == language


def length_number_sentences(response: str, num_sentences: int, relation: str) -> bool:
    return _compare(count_sentences(response), relation, num_sentences)


def length_number_words(response: str, num_words: int, relation: str) -> bool:
    return _compare(count_words(response), relation, num_words)


def length_number_paragraphs(response: str, num_paragraphs: int) -> bool:
    """Paragraphs separated by the markdown divider the prompt asks for.

    The prompts say "separated with the markdown divider: ***", so the split is
    on `***` and not on a blank line. An empty leading or trailing piece is the
    divider sitting at one end and does not count; an empty one in the middle
    means two dividers in a row, which is not the requested shape.
    """
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index in (0, len(paragraphs) - 1):
                count -= 1
            else:
                return False
    return count == num_paragraphs


def length_nth_paragraph_first_word(
    response: str, num_paragraphs: int, nth_paragraph: int, first_word: str
) -> bool:
    """These prompts separate paragraphs by a blank line, not by `***`."""
    paragraphs = re.split(r"\n\n", response)
    count = sum(1 for paragraph in paragraphs if paragraph.strip())
    if count != num_paragraphs or nth_paragraph > count:
        return False
    wanted = paragraphs[nth_paragraph - 1].strip()
    if not wanted:
        return False
    word = wanted.split()[0].strip().lstrip("'\"")
    # The first word may be followed immediately by punctuation, which is not
    # part of it: `Jasper,` satisfies a first_word of `jasper`.
    word = re.split(r"[.,?!'\"]", word)[0]
    return word.lower() == first_word.lower()


def content_number_placeholders(response: str, num_placeholders: int) -> bool:
    return len(re.findall(r"\[.*?\]", response)) >= num_placeholders


def content_postscript(response: str, postscript_marker: str) -> bool:
    """A postscript, allowing the spacing a model actually writes.

    `P.S.` shows up as `P.S.`, `P. S.` and `p.s.`, and all three are the thing
    the prompt asked for.
    """
    marker = postscript_marker.strip()
    letters = [part for part in marker.rstrip(".").split(".") if part]
    pattern = r"\s*" + r"\.\s?".join(re.escape(letter) for letter in letters)
    if marker.endswith("."):
        pattern += r"\."
    return bool(re.search(pattern, response, re.IGNORECASE | re.MULTILINE))


def format_number_bullet_lists(response: str, num_bullets: int) -> bool:
    """Exactly this many markdown bullets, in either `*` or `-` form.

    `**bold**` at the start of a line is not a bullet, which is why the star
    branch requires a non-star after it.
    """
    stars = re.findall(r"^\s*\*[^\*].*$", response, re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", response, re.MULTILINE)
    return len(stars) + len(dashes) == num_bullets


def format_constrained_response(response: str) -> bool:
    return any(allowed in response.strip() for allowed in CONSTRAINED_RESPONSES)


def format_number_highlighted_sections(response: str, num_highlights: int) -> bool:
    single = re.findall(r"\*[^\n\*]*\*", response)
    double = re.findall(r"\*\*[^\n\*]*\*\*", response)
    found = sum(1 for match in single if match.strip("*").strip())
    found += sum(1 for match in double if match.strip("*").strip())
    return found >= num_highlights


def format_multiple_sections(response: str, section_spliter: str, num_sections: int) -> bool:
    pattern = r"\s?" + re.escape(section_spliter) + r"\s?\d+\s?"
    return len(re.split(pattern, response)) - 1 >= num_sections


def format_json_format(response: str) -> bool:
    body = response.strip()
    for fence in ("```json", "```Json", "```JSON", "```"):
        body = body.removeprefix(fence)
    body = body.removesuffix("```").strip()
    try:
        json.loads(body)
    except ValueError:
        return False
    return True


def format_title(response: str) -> bool:
    return any(title.strip("<>").strip() for title in re.findall(r"<<[^\n]+>>", response))


def combination_two_responses(response: str) -> bool:
    """Two different responses, separated by six asterisks."""
    parts = response.split("******")
    answers = []
    for index, part in enumerate(parts):
        if not part.strip():
            if index not in (0, len(parts) - 1):
                return False
        else:
            answers.append(part)
    return len(answers) == 2 and answers[0].strip() != answers[1].strip()


def combination_repeat_prompt(response: str, prompt_to_repeat: str) -> bool:
    return response.strip().lower().startswith(prompt_to_repeat.strip().lower())


def startend_end_checker(response: str, end_phrase: str) -> bool:
    return response.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def startend_quotation(response: str) -> bool:
    wrapped = response.strip()
    return len(wrapped) > 1 and wrapped.startswith('"') and wrapped.endswith('"')


def change_case_english_capital(response: str) -> bool:
    """All capitals, and English.

    The English half is easy to miss and it does real work: a model that
    answers a "write in all caps" prompt in transliterated Hindi has not
    followed it, and caps alone cannot tell.
    """
    detected = detect_language(response)
    return detected is None or (response.isupper() and detected == "en")


def change_case_english_lowercase(response: str) -> bool:
    detected = detect_language(response)
    return detected is None or (response.islower() and detected == "en")


def change_case_capital_word_frequency(
    response: str, capital_frequency: int, capital_relation: str
) -> bool:
    return _compare(count_capitalised_words(response), capital_relation, capital_frequency)


def punctuation_no_comma(response: str) -> bool:
    return "," not in response


VERIFIERS: dict[str, Callable[..., bool]] = {
    "change_case:capital_word_frequency": change_case_capital_word_frequency,
    "change_case:english_capital": change_case_english_capital,
    "change_case:english_lowercase": change_case_english_lowercase,
    "combination:repeat_prompt": combination_repeat_prompt,
    "combination:two_responses": combination_two_responses,
    "detectable_content:number_placeholders": content_number_placeholders,
    "detectable_content:postscript": content_postscript,
    "detectable_format:constrained_response": format_constrained_response,
    "detectable_format:json_format": format_json_format,
    "detectable_format:multiple_sections": format_multiple_sections,
    "detectable_format:number_bullet_lists": format_number_bullet_lists,
    "detectable_format:number_highlighted_sections": format_number_highlighted_sections,
    "detectable_format:title": format_title,
    "keywords:existence": keywords_existence,
    "keywords:forbidden_words": keywords_forbidden_words,
    "keywords:frequency": keywords_frequency,
    "keywords:letter_frequency": keywords_letter_frequency,
    "language:response_language": language_response_language,
    "length_constraints:nth_paragraph_first_word": length_nth_paragraph_first_word,
    "length_constraints:number_paragraphs": length_number_paragraphs,
    "length_constraints:number_sentences": length_number_sentences,
    "length_constraints:number_words": length_number_words,
    "punctuation:no_comma": punctuation_no_comma,
    "startend:end_checker": startend_end_checker,
    "startend:quotation": startend_quotation,
}


def follows(response: str, instruction_id: str, arguments: dict) -> bool:
    """Whether one response satisfies one instruction.

    An empty response follows nothing, including the instructions a program
    would otherwise call satisfied - `no_comma` is true of the empty string.
    """
    if not response.strip():
        return False
    verifier = VERIFIERS[instruction_id]
    # The data carries a null for every argument an instruction does not take.
    given = {key: value for key, value in arguments.items() if value is not None}
    return bool(verifier(response, **given))


def loose_variants(response: str) -> tuple[str, ...]:
    """The response, plus the near-misses the loose score forgives.

    A model that opens with "Sure, here is..." or closes with an offer of more
    help has followed the instruction and framed it; so has one that wrapped the
    whole answer in markdown emphasis. Dropping the first line, the last line,
    both, and the asterisks covers those without forgiving anything else.
    """
    lines = response.split("\n")
    trimmed = (
        response,
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    )
    return trimmed + tuple(variant.replace("*", "") for variant in trimmed)


@dataclass(frozen=True, slots=True)
class Row:
    key: int
    prompt: str
    instruction_ids: tuple[str, ...]
    kwargs: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class RowScore:
    key: int
    instruction_ids: tuple[str, ...]
    strict: tuple[bool, ...]
    loose: tuple[bool, ...]

    @property
    def strict_prompt(self) -> bool:
        return all(self.strict)

    @property
    def loose_prompt(self) -> bool:
        return all(self.loose)


def load_rows(path) -> list[Row]:
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        entry = json.loads(line)
        rows.append(
            Row(
                key=entry["key"],
                prompt=entry["prompt"],
                instruction_ids=tuple(entry["instruction_id_list"]),
                kwargs=tuple(entry["kwargs"]),
            )
        )
    return rows


def score_row(row: Row, response: str) -> RowScore:
    constraints = list(zip(row.instruction_ids, row.kwargs, strict=True))
    exact = tuple(follows(response, name, arguments) for name, arguments in constraints)
    variants = loose_variants(response)
    forgiving = tuple(
        followed or any(follows(variant, name, arguments) for variant in variants)
        for followed, (name, arguments) in zip(exact, constraints, strict=True)
    )
    return RowScore(key=row.key, instruction_ids=row.instruction_ids, strict=exact, loose=forgiving)


def report(scores: list[RowScore]) -> dict[str, float]:
    """The four numbers IFEval is reported with.

    Prompt level is all-or-nothing per prompt; instruction level counts each
    constraint separately, so a prompt with three constraints and two met scores
    0 and 0.67 respectively.
    """
    if not scores:
        return {}
    instructions = sum(len(score.strict) for score in scores)
    return {
        "prompt_strict": sum(score.strict_prompt for score in scores) / len(scores),
        "prompt_loose": sum(score.loose_prompt for score in scores) / len(scores),
        "instruction_strict": sum(sum(score.strict) for score in scores) / instructions,
        "instruction_loose": sum(sum(score.loose) for score in scores) / instructions,
        "prompts": len(scores),
        "instructions": instructions,
    }


def by_instruction(scores: list[RowScore]) -> dict[str, dict[str, float]]:
    """Strict accuracy per instruction id, for seeing which rule a model breaks."""
    tally: dict[str, list[bool]] = {}
    for score in scores:
        for instruction_id, followed in zip(score.instruction_ids, score.strict, strict=True):
            tally.setdefault(instruction_id, []).append(followed)
    return {
        instruction_id: {"strict": sum(hits) / len(hits), "n": len(hits)}
        for instruction_id, hits in sorted(tally.items())
    }
