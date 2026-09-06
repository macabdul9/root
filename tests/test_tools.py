import inspect
from datetime import datetime

import pytest

from root.tools import balance_parentheses, build_tools, calculate, ensure_print, today


def test_calculate_rejects_non_arithmetic():
    with pytest.raises(ValueError):
        calculate("__import__('os').system('ls')")


def test_calculate_handles_precedence():
    assert calculate("12 * (3 + 4)") == "84"


def test_read_file_cannot_escape_the_workspace(tmp_path):
    tools = build_tools(tmp_path)
    with pytest.raises(ValueError):
        tools["read_file"].run("../secrets.txt")


def test_grep_reports_path_and_line_number(tmp_path):
    (tmp_path / "notes.txt").write_text("alpha\nbeta gamma\n")
    matches = build_tools(tmp_path)["grep"].run("gamma")
    assert matches.startswith("notes.txt:2:")


def test_grep_skips_the_virtualenv(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "vendored.py").write_text("needle\n")
    assert build_tools(tmp_path)["grep"].run("needle").startswith("no matches")


def test_json_get_walks_lists_and_objects(tmp_path):
    (tmp_path / "run.json").write_text('{"calls": [{"name": "grep"}]}')
    assert build_tools(tmp_path)["json_get"].run("run.json:calls.0.name") == "grep"


def test_json_get_rejects_a_query_without_a_path(tmp_path):
    with pytest.raises(ValueError, match="dotted.path"):
        build_tools(tmp_path)["json_get"].run("run.json")


def test_file_info_counts_lines(tmp_path):
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n")
    assert "4 lines" in build_tools(tmp_path)["file_info"].run("notes.txt")


def test_today_applies_an_offset():
    now = datetime.now()
    assert today("").startswith(now.strftime("%Y-%m-%d"))
    assert today("+400 days") > today("")


def test_today_rejects_an_unparseable_offset():
    with pytest.raises(ValueError, match="offset"):
        today("next tuesday")


def test_ensure_print_wraps_a_bare_expression():
    assert ensure_print("sum(range(1, 21))") == "print(sum(range(1, 21)))"


def test_ensure_print_leaves_a_real_program_alone():
    program = "for i in range(3):\n    print(i)"
    assert ensure_print(program) == program


def test_run_python_returns_stdout(tmp_path):
    assert build_tools(tmp_path)["run_python"].run("print(2 ** 16)") == "65536"


def test_run_python_reports_a_traceback_instead_of_raising(tmp_path):
    output = build_tools(tmp_path)["run_python"].run("print(1 / 0)")
    assert "ZeroDivisionError" in output
    assert output.startswith("error:")


def test_run_python_demands_printed_output(tmp_path):
    with pytest.raises(ValueError, match="print"):
        build_tools(tmp_path)["run_python"].run("x = 1\ny = 2")


def test_run_python_marks_a_crashed_program_as_an_error(tmp_path):
    output = build_tools(tmp_path)["run_python"].run("print(undefined_name)")
    assert output.startswith("error:")
    assert "NameError" in output


def test_balance_parentheses_drops_a_trailing_extra():
    assert balance_parentheses("sorted([2, 3, 41]))") == "sorted([2, 3, 41])"


def test_balance_parentheses_leaves_valid_code_alone():
    assert balance_parentheses("print(len('abc'))") == "print(len('abc'))"


def test_balance_parentheses_gives_up_on_other_syntax_errors():
    broken = "for i in range(3)"
    assert balance_parentheses(broken) == broken


def test_run_python_recovers_from_an_extra_closing_paren(tmp_path):
    assert build_tools(tmp_path)["run_python"].run("sorted([2,3,41,0,19,-10]))") == (
        "[-10, 0, 2, 3, 19, 41]"
    )


def test_list_files_treats_a_directory_name_as_its_contents(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.sh").write_text("")
    (tmp_path / "top.txt").write_text("")

    listing = build_tools(tmp_path)["list_files"].run("scripts")

    assert listing == "scripts/run.sh"


def test_list_files_accepts_a_dot_for_the_workspace_root(tmp_path):
    (tmp_path / "top.txt").write_text("")
    assert build_tools(tmp_path)["list_files"].run(".") == "top.txt"


def test_list_files_rejects_a_pattern_leaving_the_workspace(tmp_path):
    with pytest.raises(ValueError, match="outside the workspace"):
        build_tools(tmp_path)["list_files"].run("../*")


def test_list_files_reports_an_unusable_pattern(tmp_path):
    """Path.glob raises NotImplementedError here, which would otherwise reach
    the model as a traceback instead of a usable message."""
    with pytest.raises(ValueError, match="not a usable glob"):
        build_tools(tmp_path)["list_files"].run("///")


def test_list_files_handles_a_redundant_dot_path(tmp_path):
    (tmp_path / "top.txt").write_text("")
    assert build_tools(tmp_path)["list_files"].run("./.") == "top.txt"


def test_write_file_creates_parents_and_reports(tmp_path):
    tools = build_tools(tmp_path)
    message = tools["write_file"].run(path="notes/todo.md", content="- ship it\n")

    assert (tmp_path / "notes" / "todo.md").read_text() == "- ship it\n"
    assert message.startswith("wrote notes/todo.md")


def test_write_file_says_when_it_overwrites(tmp_path):
    tools = build_tools(tmp_path)
    tools["write_file"].run(path="a.txt", content="one")
    assert tools["write_file"].run(path="a.txt", content="two").startswith("overwrote")
    assert (tmp_path / "a.txt").read_text() == "two"


def test_write_file_cannot_escape_the_workspace(tmp_path):
    with pytest.raises(ValueError, match="outside the workspace"):
        build_tools(tmp_path)["write_file"].run(path="../escaped.txt", content="x")


def test_write_file_refuses_a_directory(tmp_path):
    (tmp_path / "somewhere").mkdir()
    with pytest.raises(IsADirectoryError):
        build_tools(tmp_path)["write_file"].run(path="somewhere", content="x")


def test_make_directory_is_repeatable(tmp_path):
    tools = build_tools(tmp_path)
    assert tools["make_directory"].run("src/models").endswith("created")
    assert tools["make_directory"].run("src/models").endswith("already exists")
    assert (tmp_path / "src" / "models").is_dir()


def test_binding_matches_named_arguments(tmp_path):
    tool = build_tools(tmp_path)["write_file"]
    assert tool.bind([("content", "hi"), ("path", "a.txt")]) == {"path": "a.txt", "content": "hi"}


def test_binding_fills_unnamed_arguments_in_order(tmp_path):
    tool = build_tools(tmp_path)["write_file"]
    assert tool.bind([(None, "a.txt"), (None, "hi")]) == {"path": "a.txt", "content": "hi"}


def test_binding_reports_a_missing_argument(tmp_path):
    tool = build_tools(tmp_path)["write_file"]
    with pytest.raises(ValueError, match="content"):
        tool.bind([("path", "a.txt")])


def test_every_tool_accepts_the_parameters_it_declares(tmp_path):
    """The schema names the model sees are the names the function is called
    with, so a mismatch reaches the model as a TypeError on every call."""
    for tool in build_tools(tmp_path).values():
        accepted = set(inspect.signature(tool.run).parameters)
        declared = {parameter.name for parameter in tool.parameters}
        assert declared == accepted, tool.name
