import pytest

from root.route import AUTO, choose_agent, is_conversational, is_identity_question

AGENTS = {
    "chat",
    "calc",
    "python",
    "files",
    "search",
    "extract",
    "code",
    "write",
    "now",
    "pdf",
    "repo",
}

# Prompts the rules were written against.
TUNED = [
    ("who are you?", {"chat"}),
    ("what is 2+3?", {"calc", "python"}),
    ("[2,3,41,0,19,-10] sort this number", {"python"}),
    ("What Python version does pyproject.toml require?", {"files"}),
    ("Which file defines the normalize_scores function?", {"search", "files"}),
    ("Invoice from Dana Ruiz, dated 2026-03-14, for 412.50 USD of GPU rental.", {"extract"}),
    ("What is the sum of the squares of the numbers 1 through 20?", {"python", "calc"}),
    ("In one sentence, what is a KV cache?", {"chat"}),
    ("Which files are in the scripts directory?", {"files"}),
    ("How many days are there between 2026-01-01 and 2026-09-06?", {"python"}),
    ("What is 17 times 23, plus 4?", {"calc", "python"}),
    ("A box holds 24 pens. I buy 7 boxes and give away 13 pens. How many are left?", {"calc"}),
    ("grep for force_first_call", {"search", "files"}),
    ("Read notes.md and tell me what it says", {"files"}),
    ("Receipt: 2026-02-02, 18.40 EUR, taxi", {"extract"}),
    ("Who wrote the Iliad?", {"chat"}),
]

# Written after the rules, to measure how they generalise.
HELD_OUT = [
    ("What is 8 to the power of 5?", {"calc", "python"}),
    ("Tell me about the Bronze Age collapse", {"chat"}),
    ("Which script launches the evaluation?", {"search", "files"}),
    ("Trim the whitespace from ' padded '", {"python"}),
    ("What day of the week was 1999-12-31?", {"python"}),
    ("Order confirmation 2026-06-01 for 89.99 GBP from Kite Ltd", {"extract"}),
    ("Do you think Python is a good first language?", {"chat"}),
    ("How big is the configs folder?", {"files"}),
    ("Take 12 percent off 340", {"calc", "python"}),
    ("Give me the unique items in [4, 4, 9, 1, 9]", {"python"}),
    ("Why does bfloat16 lose precision?", {"chat"}),
    ("Print the first 20 Fibonacci numbers", {"python"}),
    ("What does the README say about evals?", {"files"}),
    ("Multiply 340 by 1.15", {"calc", "python"}),
    ("Name three seabirds", {"chat"}),
    ("Where do we set MAX_TOOL_OUTPUT_CHARS?", {"search", "files"}),
]


def test_no_prompt_is_sent_to_the_wrong_agent():
    """The safety property: a rule either fires correctly or does not fire.

    A miss costs nothing - the default agent answers, as it would have anyway.
    A wrong route runs the wrong tool, so any regression here matters more than
    a drop in coverage.
    """
    misrouted = [
        (prompt, choose_agent(prompt, AGENTS).agent)
        for prompt, acceptable in TUNED + HELD_OUT
        if choose_agent(prompt, AGENTS).agent not in acceptable | {"chat"}
    ]
    assert misrouted == []


def test_held_out_coverage_does_not_regress():
    hits = sum(choose_agent(p, AGENTS).agent in ok for p, ok in HELD_OUT)
    assert hits >= 9, f"routed {hits}/{len(HELD_OUT)} held-out prompts"


def test_tuned_prompts_all_route():
    misses = [p for p, ok in TUNED if choose_agent(p, AGENTS).agent not in ok]
    assert misses == []


def test_an_unmatched_prompt_falls_through_to_the_default():
    route = choose_agent("Name three seabirds", AGENTS)
    assert route.agent == "chat"
    assert route.rule is None
    assert not route.matched


def test_a_record_beats_the_bare_date_rule():
    assert choose_agent("Invoice 2026-03-14 for 412.50 USD", AGENTS).agent == "extract"
    assert choose_agent("What happened on 2026-03-14?", AGENTS).agent == "python"


def test_only_configured_agents_are_chosen():
    route = choose_agent("[1, 2, 3] sort this", {"chat", "calc"})
    assert route.agent == "chat"


def test_the_default_falls_back_when_it_is_not_configured():
    assert choose_agent("hello there", {"calc"}).agent == "calc"


def test_auto_is_not_a_routable_target():
    assert choose_agent("what is 2+2", AGENTS).agent != AUTO


CONVERSATIONAL = [
    "what can you do?",
    "who are you?",
    "what are you",
    "how do you work?",
    "what tools do you have?",
    "hi",
    "thanks",
    "sure",
    "yes",
]

TASKS = [
    "sort [3, 1, 2]",
    "can you sort this list [3, 1, 2]",
    "What Python version does pyproject.toml require?",
    "What is 17 times 23?",
    "yesterday's date",
]


@pytest.mark.parametrize("prompt", CONVERSATIONAL)
def test_a_question_about_the_assistant_is_conversational(prompt):
    assert is_conversational(prompt)


@pytest.mark.parametrize("prompt", TASKS)
def test_a_task_is_not_conversational(prompt):
    assert not is_conversational(prompt)


def test_writing_code_is_not_running_code():
    assert choose_agent("write python code for binary search", AGENTS).agent == "code"
    assert choose_agent("implement a function to reverse a list", AGENTS).agent == "code"


def test_running_something_still_goes_to_python():
    assert choose_agent("What is the sum of squares of 1 through 20?", AGENTS).agent != "code"
    assert choose_agent("[3, 1, 2] sort this", AGENTS).agent == "python"


def test_creating_a_file_goes_to_the_agent_that_can():
    for prompt in ("create a file called notes.txt", "can you create files?", "write a file"):
        assert choose_agent(prompt, AGENTS).agent == "write", prompt


def test_asking_the_time_reaches_the_clock():
    for prompt in ("what time it is?", "what time is it", "what is today's date?"):
        assert choose_agent(prompt, AGENTS).agent == "now", prompt


def test_date_arithmetic_is_not_the_clock():
    assert (
        choose_agent("How many days between 2026-01-01 and 2026-09-06?", AGENTS).agent == "python"
    )
    assert choose_agent("What day of the week was 1999-12-31?", AGENTS).agent == "python"


def test_reading_a_file_still_goes_to_files():
    assert choose_agent("What Python version does pyproject.toml require?", AGENTS).agent == "files"


def test_a_pdf_question_reaches_the_pdf_agent():
    for prompt in ("What does report.pdf say?", "summarise the pdf"):
        assert choose_agent(prompt, AGENTS).agent == "pdf", prompt


def test_a_repository_question_reaches_the_repo_agent():
    for prompt in ("what changed in the working tree?", "show me the last 5 commits"):
        assert choose_agent(prompt, AGENTS).agent == "repo", prompt


WRITING = [
    "Create a file called notes.txt",
    "Append the line done to log.txt",
    "Add a line to log.txt",
    "Rename log.txt to history.txt",
    "Move report.pdf into archive",
    "Delete scratch.txt",
    "Save the output to results.csv",
    "make a directory called notes",
]

READING = [
    "What Python version does project.toml require?",
    "Read notes.md and tell me what it says",
    "Show me lines 1 to 5 of src/model.py",
    "What does README.md say about evals?",
]


@pytest.mark.parametrize("prompt", WRITING)
def test_changing_a_file_reaches_the_write_agent(prompt):
    assert choose_agent(prompt, AGENTS).agent == "write", prompt


@pytest.mark.parametrize("prompt", READING)
def test_reading_a_file_still_reaches_the_files_agent(prompt):
    assert choose_agent(prompt, AGENTS).agent == "files", prompt


@pytest.mark.parametrize(
    "prompt",
    [
        "What is 5 plus 10?",
        "Remove the duplicates from [1, 2, 2, 3]",
        "write python code for binary search",
    ],
)
def test_a_write_verb_without_a_file_does_not_reach_the_write_agent(prompt):
    assert choose_agent(prompt, AGENTS).agent != "write", prompt


IDENTITY = [
    "who are you?",
    "who are you",
    "Who r u",
    "what are you?",
    "what's your name?",
    "what is your name",
    "who built you?",
    "who made this?",
    "what model are you?",
    "which model is this?",
    "introduce yourself",
    "tell me about yourself",
]

NOT_IDENTITY = [
    "who wrote the Iliad?",
    "what model of car is a Corolla?",
    "what are you going to do with [1,2,3]?",
    "who is the author of pipeline.py?",
    "what is 2+2",
]


@pytest.mark.parametrize("prompt", IDENTITY)
def test_identity_questions_are_recognised(prompt):
    assert is_identity_question(prompt)


@pytest.mark.parametrize("prompt", NOT_IDENTITY)
def test_other_questions_are_not_identity(prompt):
    assert not is_identity_question(prompt)
