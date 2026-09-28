"""cli.py must not crash when a live model prints non-ASCII into a pipe.

A Windows pipe or file redirect encodes stdout as cp1252. A live Claude or Gemini
worker writes characters such as an arrow, and printing one used to raise
UnicodeEncodeError mid-stream. The stub worker prints only ASCII, so no other test
reaches this.
"""

import io
import sys

import cli

ARROW = chr(0x2192)
NEWLINE = chr(10)


def test_utf8_output_survives_a_cp1252_pipe(monkeypatch):
    raw = io.BytesIO()
    pipe = io.TextIOWrapper(raw, encoding="cp1252", newline=NEWLINE)
    monkeypatch.setattr(sys, "stdout", pipe)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    cli._utf8_output()
    print(f"Fixed both calls {ARROW} tests pass")
    sys.stdout.flush()

    assert raw.getvalue().decode("utf-8") == f"Fixed both calls {ARROW} tests pass{NEWLINE}"


def test_cp1252_pipe_really_fails_without_it():
    pipe = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    try:
        pipe.write(ARROW)
        pipe.flush()
    except UnicodeEncodeError:
        return
    raise AssertionError("expected cp1252 to reject an arrow; the test above would prove nothing")
