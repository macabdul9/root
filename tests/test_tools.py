from datetime import datetime

import pytest

from root.tools import build_tools, calculate, ensure_print, today


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
