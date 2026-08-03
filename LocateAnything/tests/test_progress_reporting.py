from __future__ import annotations

import io

from compiler.scripts.common.progress import resolve_progress_mode, track


def test_log_progress_reports_count_rate_and_eta(monkeypatch):
    monkeypatch.setenv("LA_PROGRESS", "log")
    stream = io.StringIO()

    reporter = track(["a", "b"], "Prepare", unit="sample", stream=stream)
    assert list(reporter) == ["a", "b"]

    output = stream.getvalue()
    assert "[progress] Prepare: 2/2 sample (100.0%)" in output
    assert "elapsed=" in output
    assert "rate=" in output
    assert "eta=" in output


def test_off_progress_is_silent(monkeypatch):
    monkeypatch.setenv("LA_PROGRESS", "off")
    stream = io.StringIO()

    assert list(track([1, 2, 3], "Build", stream=stream)) == [1, 2, 3]
    assert stream.getvalue() == ""


def test_auto_progress_uses_log_for_non_terminal_stream(monkeypatch):
    monkeypatch.delenv("LA_PROGRESS", raising=False)
    assert resolve_progress_mode(stream=io.StringIO()) == "log"
