"""Regression tests for subprocess handling in :mod:`rivia.model._mapper`.

These cover a two-pipe deadlock in the streaming branch of ``_run_subprocess``:
stdout was drained to EOF before stderr was read at all, so a child that filled
the stderr pipe buffer blocked forever, and the ``timeout`` argument -- applied
only at ``proc.wait()``, after both drain loops -- could never fire.

No HEC-RAS install or fixture model is needed; any chatty child reproduces it.
"""

import locale
import os
import signal
import subprocess
import sys
import time
import warnings
from contextlib import suppress
from pathlib import Path

import pytest

import rivia.model._mapper as mapper
from rivia.model._mapper import _run_subprocess

SUBPROC_TIMEOUT = 10  # what _run_subprocess is asked to enforce
PYTEST_TIMEOUT = 30  # outer safety net; must stay well above the above

# 0x81 and 0x8D are undefined in cp1252 and invalid as UTF-8 -- but NOT in every
# possible locale codec: latin-1 and cp437 decode all 256 byte values.  The
# encoding test below skips itself rather than asserting a behaviour the
# platform cannot exhibit.
UNDECODABLE = b"\x81\x8d"


def _locale_codec_rejects(data: bytes) -> bool:
    """True when the codec that ``text=True`` selects would raise on ``data``."""
    try:
        data.decode(locale.getpreferredencoding(False))
    except UnicodeDecodeError:
        return True
    return False


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill_recorded_pid(pid_file: Path) -> None:
    """Kill the recorded sleeper, then verify it actually died.

    Best-effort by nature -- it cannot guarantee termination -- so it warns
    rather than failing the test if the process survives.  A warning on the
    runner is what tells you a sleeper leaked; silence would not.
    """
    if not pid_file.exists():
        return
    pid: int | None = None
    with suppress(Exception):
        pid = int(pid_file.read_text().strip())
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                check=False,
                timeout=10,
            )
        else:
            os.kill(pid, signal.SIGKILL)
    if pid is None:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            alive = _pid_alive(pid)
        except Exception as exc:  # noqa: BLE001 - cleanup must not fail a test
            # _pid_alive shells out to tasklist, which can fail for reasons
            # unrelated to the test.  A cleanup helper that raises would fail
            # or mask the test it is cleaning up after.
            warnings.warn(
                f"test cleanup could not check PID {pid}: {exc}", stacklevel=2
            )
            return
        if not alive:
            return
        time.sleep(0.1)
    warnings.warn(
        f"test cleanup could not kill PID {pid}; it may linger", stacklevel=2
    )


@pytest.mark.timeout(PYTEST_TIMEOUT)
def test_stream_output_survives_stderr_larger_than_pipe_buffer() -> None:
    """Draining stdout to EOF before reading stderr deadlocks once the child fills
    the stderr pipe buffer (4-64 KB on Windows). 500 KB is comfortably past it."""
    code = (
        "import sys\n"
        "sys.stdout.write('DEBUG: start\\n'); sys.stdout.flush()\n"
        "sys.stderr.write('x' * 500_000); sys.stderr.flush()\n"
        "sys.stdout.write('INFO: done\\n'); sys.stdout.flush()\n"
    )
    result = _run_subprocess([sys.executable, "-c", code], None, SUBPROC_TIMEOUT, True)

    assert result.returncode == 0
    assert "INFO: done" in result.stdout  # proves stdout drained past the block
    assert len(result.stderr) >= 500_000  # proves stderr was drained too


@pytest.mark.timeout(PYTEST_TIMEOUT)
def test_stream_output_timeout_actually_fires() -> None:
    """timeout was applied only at proc.wait(), after the drain loops, so on the
    streaming path it could never fire."""
    code = "import time\ntime.sleep(120)\n"
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_subprocess([sys.executable, "-c", code], None, 3, True)
    elapsed = time.monotonic() - start
    # The timeout must fire on its own schedule, not be rescued by pytest's.
    assert elapsed < PYTEST_TIMEOUT / 2, f"timeout took {elapsed:.1f}s"


@pytest.mark.skipif(
    not _locale_codec_rejects(UNDECODABLE),
    reason=(
        f"locale codec {locale.getpreferredencoding(False)!r} accepts this "
        f"fixture, so errors='replace' is unobservable with it"
    ),
)
@pytest.mark.timeout(PYTEST_TIMEOUT)
def test_stream_output_decodes_undecodable_bytes_without_losing_the_stream() -> None:
    """errors='replace' keeps a pump alive through output the locale codec rejects.

    (b'\\xff\\xfe' would NOT work as a fixture on Windows: cp1252 decodes both
    bytes happily as U+00FF U+00FE.)
    """
    code = (
        "import sys\n"
        "sys.stderr.buffer.write(b'\\x81\\x8d bad bytes\\n')\n"
        "sys.stderr.buffer.write(b'STDERR-SENTINEL\\n'); sys.stderr.flush()\n"
        "sys.stdout.write('INFO: done\\n'); sys.stdout.flush()\n"
    )
    result = _run_subprocess([sys.executable, "-c", code], None, SUBPROC_TIMEOUT, True)

    assert result.returncode == 0
    assert "INFO: done" in result.stdout
    assert "bad bytes" in result.stderr
    # The sentinel is the real assertion: it proves the reader kept going
    # *past* the undecodable bytes rather than dying at them.
    assert "STDERR-SENTINEL" in result.stderr


@pytest.mark.timeout(PYTEST_TIMEOUT)
def test_non_streaming_branch_accepts_cwd_none() -> None:
    """cwd=str(cwd) turned None into the literal directory name 'None'."""
    result = _run_subprocess(
        [sys.executable, "-c", "print('ok')"], None, SUBPROC_TIMEOUT, False
    )
    assert result.returncode == 0
    assert "ok" in result.stdout


@pytest.mark.timeout(PYTEST_TIMEOUT)
def test_reader_failure_raises_instead_of_returning_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader that dies must not yield a CompletedProcess with partial stderr.

    The caller keys retry and failure on stderr substrings, so partial stderr
    turns a real failure into apparent success.
    """

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated reader failure")

    monkeypatch.setattr(mapper.logger, "warning", boom)
    code = "import sys; sys.stderr.write('Exception: real failure\\n')"

    with pytest.raises(RuntimeError, match="output reader failed"):
        _run_subprocess([sys.executable, "-c", code], None, SUBPROC_TIMEOUT, True)


@pytest.mark.timeout(PYTEST_TIMEOUT)
def test_stuck_reader_is_bounded_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A descendant that inherits the pipe keeps it open after the child exits.

    proc.wait() returns, but the readers never see EOF.  The call must raise
    within the grace window rather than block -- and must NOT return a partial
    CompletedProcess.  _READER_GRACE is shortened so this costs ~1 s, not 10.
    """
    monkeypatch.setattr(mapper, "_READER_GRACE", 0.5)
    pid_file = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "                     stdout=sys.stdout, stderr=sys.stderr)\n"
        f"open(r'{pid_file}', 'w').write(str(p.pid))\n"
        "sys.stdout.write('INFO: parent done\\n'); sys.stdout.flush()\n"
    )
    try:
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="never reached EOF"):
            _run_subprocess([sys.executable, "-c", code], None, SUBPROC_TIMEOUT, True)
        elapsed = time.monotonic() - start
        assert elapsed < 10, f"teardown took {elapsed:.1f}s"
    finally:
        _kill_recorded_pid(pid_file)


@pytest.mark.skipif(sys.platform != "win32", reason="taskkill /T is Windows-only")
@pytest.mark.timeout(PYTEST_TIMEOUT)
def test_timeout_kills_the_whole_process_tree(tmp_path: Path) -> None:
    """On the timeout path the root is still alive, so taskkill /T can walk it.

    This is the one path where descendant cleanup is actually claimed, so it is
    the one path that gets a positive assertion: the grandchild must die.
    """
    pid_file = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open(r'{pid_file}', 'w').write(str(p.pid))\n"
        "time.sleep(120)\n"  # root stays alive, so the timeout is what fires
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_subprocess([sys.executable, "-c", code], None, 3, True)

        pid = int(pid_file.read_text().strip())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _pid_alive(pid):
            time.sleep(0.1)
        # Unguarded on purpose: unlike the cleanup helper, being unable to tell
        # whether the grandchild died should fail this test, not pass quietly.
        assert not _pid_alive(pid), f"grandchild {pid} survived the tree kill"
    finally:
        _kill_recorded_pid(pid_file)
