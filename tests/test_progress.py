import io

from root.progress import FRAMES, working


class FakeTerminal(io.StringIO):
    def isatty(self):
        return True


def test_a_pipe_gets_one_static_line():
    """Animation into a log file is just carriage returns."""
    stream = io.StringIO()
    with working("Loading something", stream=stream):
        pass

    assert stream.getvalue() == "Loading something...\n"


def test_a_terminal_gets_frames_and_a_clean_line_afterwards():
    stream = FakeTerminal()
    with working("Loading", stream=stream):
        pass
    written = stream.getvalue()

    assert written.endswith("\r")
    assert "Loading" not in written.rsplit("\r", 2)[-1]


def test_the_message_is_cleared_even_when_the_work_raises():
    stream = FakeTerminal()
    try:
        with working("Loading", stream=stream):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert stream.getvalue().endswith("\r")


def test_frames_move():
    assert len(set(FRAMES)) == len(FRAMES)
    assert FRAMES[0].strip() == ""
    assert FRAMES[-1] == "..."
