import pytest

from filinglens.config import Settings


def test_csv_env_values_become_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORMS", "10-K, 10-Q,10-K/A")
    monkeypatch.setenv("API_CORS_ORIGINS", "http://a.test,http://b.test")
    settings = Settings(_env_file=None)
    assert settings.forms == ["10-K", "10-Q", "10-K/A"]
    assert settings.api_cors_origins == ["http://a.test", "http://b.test"]


def test_rate_limit_cannot_exceed_sec_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGAR_MAX_RPS", "12")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_missing_edgar_identity_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    with pytest.raises(ValueError, match="EDGAR_IDENTITY"):
        Settings(_env_file=None).require_edgar_identity()
