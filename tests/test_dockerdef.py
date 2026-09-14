import pytest

from root.dockerdef import Unsupported, flatten, instructions, translate


def build(body, context="/ctx"):
    return translate(body, context)


def test_a_wrapped_run_keeps_its_continuations():
    """Dropping the backslashes turns one `pip install` into a dozen bare
    package names the shell tries to execute."""
    written = build("FROM python:3.12\nRUN pip install \\\n    numpy \\\n    scipy\n")
    assert "\\\n" in written
    assert "numpy" in written and "scipy" in written


def test_a_cd_inside_one_run_does_not_leak_into_the_next():
    """Docker starts every RUN at WORKDIR. A shared %post shell would carry the
    cd forward and silently run later commands in the wrong directory."""
    written = build("FROM x\nWORKDIR /app\nRUN cd /tmp && touch a\nRUN touch b\n")
    runs = [line for line in written.splitlines() if line.strip().startswith("(")]
    assert len(runs) == 2
    assert all("cd /app;" in run for run in runs)


def test_copy_and_run_keep_their_original_order():
    """This Dockerfile copies a checksum file, verifies it, then copies more.
    Hoisting every COPY above every RUN is a different build."""
    written = build(
        "FROM x\nCOPY sums /app/sums\nRUN sha256sum -c /app/sums\nCOPY late /app/late\n"
    )
    body = written[written.index("%post") :]
    assert body.index("/app/sums") < body.index("sha256sum") < body.index("/app/late")


def test_a_relative_copy_destination_resolves_against_workdir():
    written = build("FROM x\nWORKDIR /srv\nCOPY thing data/thing\n")
    assert "/srv/data/thing" in written


def test_a_directory_copy_takes_the_contents_like_docker_does():
    """`cp -r src dst` copies the directory itself when dst exists, which is not
    what COPY does."""
    written = build("FROM x\nCOPY data /app/data\n")
    assert "/.'" in written or "/." in written


def test_env_is_exported_for_later_build_steps_and_for_the_container():
    written = build("FROM x\nENV KEY=value\nRUN echo $KEY\n")
    assert written.count("export KEY=value") == 2
    assert "%environment" in written


def test_the_two_env_spellings_are_both_read():
    assert "export A=b" in build("FROM x\nENV A=b\n")
    assert "export A=b" in build("FROM x\nENV A b\n")


def test_workdir_is_created_not_only_entered():
    assert "mkdir -p /app" in build("FROM x\nWORKDIR /app\n")


def test_the_build_context_is_staged_and_then_removed():
    written = build("FROM x\nCOPY a /a\n", context="/some/ctx")
    assert "/some/ctx /docker-context" in written
    assert written.index("rm -rf /docker-context") > written.index("/docker-context/a")


def test_a_multi_stage_build_is_refused_rather_than_approximated():
    """Seven of the benchmark's tasks use one; a wrong image is worse than a
    missing one, because it scores."""
    with pytest.raises(Unsupported, match="multi-stage"):
        build("FROM python:3.12 AS builder\nRUN touch a\nFROM python:3.12\n")


def test_copy_from_another_stage_is_refused():
    with pytest.raises(Unsupported, match="multi-stage"):
        build("FROM x\nCOPY --from=builder /a /b\n")


def test_an_untranslated_instruction_is_refused():
    with pytest.raises(Unsupported, match="VOLUME"):
        build("FROM x\nVOLUME /data\n")


def test_comments_and_blank_lines_are_dropped():
    parsed = instructions("# canary\n\nFROM x\n# note\nRUN true\n")
    assert [item.keyword for item in parsed] == ["FROM", "RUN"]


def test_a_digest_pinned_base_survives_translation():
    written = build("FROM python:3.12-slim@sha256:abc123\n")
    assert "From: python:3.12-slim@sha256:abc123" in written


def test_flatten_joins_a_wrapped_word_instruction():
    assert flatten("a \\\n    b") == "a b"
