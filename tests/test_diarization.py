"""Unit tests for app.diarization — factory, stubs, and no-op."""

from pathlib import Path

import pytest

from app.diarization import (
    Diarizer,
    HostedDiarizerStub,
    NoopDiarizer,
    SpeakerTurn,
    get_diarizer,
)


class TestSpeakerTurn:
    def test_dataclass_fields(self):
        turn = SpeakerTurn(start_sec=1.0, end_sec=2.5, speaker_label="SPEAKER_00")
        assert turn.start_sec == 1.0
        assert turn.end_sec == 2.5
        assert turn.speaker_label == "SPEAKER_00"

    def test_frozen(self):
        turn = SpeakerTurn(0, 1, "A")
        with pytest.raises(AttributeError):
            turn.start_sec = 99  # frozen dataclass


class TestNoopDiarizer:
    def test_returns_empty_list(self):
        diarizer = NoopDiarizer()
        assert diarizer.diarize(Path("/fake/path.wav")) == []

    def test_is_diarizer_subclass(self):
        assert isinstance(NoopDiarizer(), Diarizer)


class TestHostedDiarizerStub:
    def test_raises_runtime_error(self):
        stub = HostedDiarizerStub()
        with pytest.raises(RuntimeError, match="not configured"):
            stub.diarize(Path("/fake/path.wav"))


class TestGetDiarizer:
    def test_local_no_token_returns_noop(self, monkeypatch):
        monkeypatch.setattr("app.diarization.settings.hf_token", None)
        monkeypatch.setattr("app.diarization.settings.diarizer_provider", "local")
        result = get_diarizer()
        assert isinstance(result, NoopDiarizer)

    def test_local_with_token_returns_local(self, monkeypatch):
        monkeypatch.setattr("app.diarization.settings.hf_token", "dummy_token")
        monkeypatch.setattr("app.diarization.settings.diarizer_provider", "local")
        result = get_diarizer()
        from app.diarization import LocalPyannoteDializer

        assert isinstance(result, LocalPyannoteDializer)

    def test_hosted_returns_stub(self, monkeypatch):
        monkeypatch.setattr("app.diarization.settings.diarizer_provider", "hosted")
        result = get_diarizer()
        assert isinstance(result, HostedDiarizerStub)

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setattr("app.diarization.settings.diarizer_provider", "nonexistent")
        with pytest.raises(ValueError, match="Unknown diarizer provider"):
            get_diarizer()
